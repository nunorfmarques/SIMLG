import simpy
import matplotlib.pyplot as plt
import numpy as np
import time
import streamlit as st
import json
import streamlit.components.v1 as components
import random

# ==========================================
# TOPOLOGIA E COORDENADAS FÍSICAS (Em Metros)
# ==========================================
# Definição dos nós principais de navegação (X, Y)
NOS_MAPA = {
    "PORTARIA_ENTRADA": np.array([0, 0]),
    "ENTRADA_PARQUE":   np.array([40, 0]),
    "SAIDA_PARQUE":     np.array([40, 30]),
    "EIXO_MANOBRAS":    np.array([100, 0]),   # Corredor principal em frente aos cais
    "PORTARIA_SAIDA":   np.array([0, -15])
}

# Dimensões físicas dos recursos (Comprimento x Largura)
DIM_CAMIAO = (16.5, 2.5) # Camião articulado padrão (TIR)
DIM_CAIS = (20.0, 4.0)   # Espaço físico de cada cais
DIM_LUGAR = (18.0, 3.5)  # Espaço de cada lugar no parque de espera


# ==========================================
# LÓGICA BASE DO MOTOR DE SIMULAÇÃO
# ==========================================
def processo_viatura(env, id_viatura, cais, params, dados):
    """
    Processo SimPy que representa o ciclo de vida de uma viatura.
    """
    t_chegada = env.now

    dados["historico_fila"].append((env.now, len(cais.queue)))

    with cais.request() as pedido:
        yield pedido

        t_inicio_carga = env.now
        espera = t_inicio_carga - t_chegada

        dados["historico_ocupacao"].append((env.now, cais.count))

        yield env.timeout(params["tempo_carga"])

        t_fim_carga = env.now

        dados["historico_ocupacao"].append((env.now, cais.count - 1))
        dados["historico_fila"].append((env.now, len(cais.queue)))

        dados["registos"].append({
            "viatura":        id_viatura,
            "t_chegada":      t_chegada,
            "t_inicio_carga": t_inicio_carga,
            "t_fim_carga":    t_fim_carga,
            "espera":         espera,
            "tempo_carga":    params["tempo_carga"],
        })

def gerador_chegadas(env, cais, params, dados):
    """
    Gera chegadas de viaturas segundo uma distribuição exponencial.
    O 'intervalo_chegada' atua agora como a média de tempo entre chegadas.
    """
    i = 0
    while True:
        # Cálculo estocástico: converte a média no tempo exato da próxima chegada
        intervalo_real = random.expovariate(1.0 / params["intervalo_chegada"])
        yield env.timeout(intervalo_real)
        
        i += 1
        env.process(processo_viatura(env, f"V{i:02d}", cais, params, dados))


def executar_simulacao(params):
    """
    Cria o ambiente SimPy, executa a simulação e devolve os dados recolhidos.
    """
    dados = {
        "registos":           [],
        "historico_fila":     [],
        "historico_ocupacao": [],
    }

    env = simpy.Environment()
    cais = simpy.Resource(env, capacity=params["num_cais"])

    env.process(gerador_chegadas(env, cais, params, dados))
    env.run(until=params["duracao_sim"])

    return dados

def calcular_indicadores(dados, params):
    """
    Calcula os indicadores de desempenho a partir dos dados recolhidos.
    """
    registos = dados["registos"]
    if not registos:
        return {
            "total_viaturas": 0, "espera_media": 0, "espera_max": 0, 
            "espera_min": 0, "taxa_ocupacao_pct": 0, "tamanho_max_fila": 0
        }

    esperas = [r["espera"] for r in registos]

    if dados["historico_ocupacao"]:
        tempos_oc = [d[0] for d in dados["historico_ocupacao"]]
        valores_oc = [d[1] for d in dados["historico_ocupacao"]]
        
        # ALTERAÇÃO: np.trapz substituído por np.trapezoid para compatibilidade com NumPy >= 2.0
        area = np.trapezoid(valores_oc, tempos_oc)
        
        taxa_ocup = (area / params["duracao_sim"]) / params["num_cais"] * 100
    else:
        taxa_ocup = 0

    if dados["historico_fila"]:
        max_fila = max(d[1] for d in dados["historico_fila"])
    else:
        max_fila = 0

    return {
        "total_viaturas":     len(registos),
        "espera_media":       np.mean(esperas),
        "espera_max":         np.max(esperas),
        "espera_min":         np.min(esperas),
        "taxa_ocupacao_pct":  round(taxa_ocup, 1),
        "tamanho_max_fila":   max_fila,
    }

def grafico_fila_e_ocupacao(dados, params, ax_fila, ax_ocup):
    """
    Plota a evolução da fila e da ocupação dos cais num par de eixos.
    """
    if dados["historico_fila"]:
        tf = [d[0] for d in dados["historico_fila"]]
        vf = [d[1] for d in dados["historico_fila"]]
        ax_fila.step(tf, vf, where="post", color="#378ADD", linewidth=1.5)
        ax_fila.fill_between(tf, vf, step="post", alpha=0.15, color="#378ADD")
    ax_fila.set_ylabel("Nº viaturas em fila")
    ax_fila.set_title("Evolução da Fila de Espera", fontsize=11, fontweight="bold")
    ax_fila.set_ylim(bottom=0)
    ax_fila.grid(True, alpha=0.3)

    if dados["historico_ocupacao"]:
        to = [d[0] for d in dados["historico_ocupacao"]]
        vo = [d[1] for d in dados["historico_ocupacao"]]
        vo_pct = [v / params["num_cais"] * 100 for v in vo]
        ax_ocup.step(to, vo_pct, where="post", color="#1D9E75", linewidth=1.5)
        ax_ocup.fill_between(to, vo_pct, step="post", alpha=0.15, color="#1D9E75")
    ax_ocup.set_ylabel("Ocupação dos cais (%)")
    ax_ocup.set_xlabel("Tempo (min)")
    ax_ocup.set_ylim(0, 110)
    ax_ocup.axhline(100, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_ocup.grid(True, alpha=0.3)


def desenhar_mapa_suave(dados, params, t_atual):
    """
    Constrói um mapa espacial 2D à escala real (Metros) com interpolação de movimento.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 1. Desenhar a Infraestrutura Físicas em Metros
    vias_x = [NOS_MAPA["PORTARIA_ENTRADA"][0], NOS_MAPA["EIXO_MANOBRAS"][0], NOS_MAPA["PORTARIA_SAIDA"][0]]
    vias_y = [NOS_MAPA["PORTARIA_ENTRADA"][1], NOS_MAPA["EIXO_MANOBRAS"][1], NOS_MAPA["PORTARIA_SAIDA"][1]]
    ax.plot(vias_x[:2], vias_y[:2], color="#E0E0E0", linewidth=20, solid_capstyle="round", zorder=0)
    ax.plot([NOS_MAPA["EIXO_MANOBRAS"][0], NOS_MAPA["PORTARIA_SAIDA"][0]], 
            [NOS_MAPA["EIXO_MANOBRAS"][1], NOS_MAPA["PORTARIA_SAIDA"][1]], color="#E0E0E0", linewidth=20, solid_capstyle="round", zorder=0)

    # Parque de Espera
    ax.add_patch(plt.Rectangle((40, 10), 40, 40, fill=True, color="#E8F0FE", edgecolor="#378ADD", alpha=0.5, zorder=1))
    ax.text(60, 52, "Parque de Espera", ha="center", fontweight="bold", fontsize=9, color="#333333")

    # Aberturas (Cais)
    posicoes_cais_y = np.linspace(10, 10 + (params["num_cais"] * 8), params["num_cais"])
    for i, y in enumerate(posicoes_cais_y):
        ax.add_patch(plt.Rectangle((NOS_MAPA["EIXO_MANOBRAS"][0], y - (DIM_CAIS[1]/2)), DIM_CAIS[0], DIM_CAIS[1], 
                                   fill=True, color="#E6F4EA", edgecolor="#1D9E75", alpha=0.8, zorder=1))
        ax.text(NOS_MAPA["EIXO_MANOBRAS"][0] + 10, y, f"Cais {i+1}", ha="center", va="center", fontsize=8, color="#333333")

    ax.text(NOS_MAPA["PORTARIA_ENTRADA"][0], NOS_MAPA["PORTARIA_ENTRADA"][1] + 5, "PORTARIA\nIN", ha="center", fontweight="bold", color="grey")
    ax.text(NOS_MAPA["PORTARIA_SAIDA"][0], NOS_MAPA["PORTARIA_SAIDA"][1] - 5, "PORTARIA\nOUT", ha="center", fontweight="bold", color="grey")

    T_MOV = 1.5 

    for r in dados["registos"]:
        t_in = r["t_chegada"]
        t_load = r["t_inicio_carga"]
        t_out = r["t_fim_carga"]

        if t_atual < t_in or t_atual > t_out + T_MOV:
            continue 

        id_num = int(r["viatura"].replace("V", ""))
        cais_idx = id_num % params["num_cais"]
        
        fila_slot = id_num % 15 
        col = fila_slot % 2
        row = fila_slot // 2
        P_FILA = np.array([45 + (col * DIM_LUGAR[0]), 45 - (row * DIM_LUGAR[1])])
        P_CAIS = np.array([NOS_MAPA["EIXO_MANOBRAS"][0] + (DIM_CAMIAO[0]/2), posicoes_cais_y[cais_idx]])

        pos = None
        cor = "#D85A30"

        if t_load == t_in: 
            if t_atual < t_in + T_MOV:
                pos = NOS_MAPA["PORTARIA_ENTRADA"] + (P_CAIS - NOS_MAPA["PORTARIA_ENTRADA"]) * ((t_atual - t_in) / T_MOV)
            elif t_atual <= t_out:
                pos = P_CAIS
                cor = "#1D9E75"
            else:
                pos = P_CAIS + (NOS_MAPA["PORTARIA_SAIDA"] - P_CAIS) * ((t_atual - t_out) / T_MOV)
        else: 
            if t_atual < t_in + T_MOV:
                pos = NOS_MAPA["PORTARIA_ENTRADA"] + (P_FILA - NOS_MAPA["PORTARIA_ENTRADA"]) * ((t_atual - t_in) / T_MOV)
            elif t_atual <= t_load:
                pos = P_FILA
            elif t_atual < t_load + T_MOV:
                pos = P_FILA + (P_CAIS - P_FILA) * ((t_atual - t_load) / T_MOV)
            elif t_atual <= t_out:
                pos = P_CAIS
                cor = "#1D9E75"
            else:
                pos = P_CAIS + (NOS_MAPA["PORTARIA_SAIDA"] - P_CAIS) * ((t_atual - t_out) / T_MOV)

        if pos is not None:
            ax.plot(pos[0], pos[1], "o", color=cor, markersize=10, zorder=3)
            ax.text(pos[0], pos[1] + 3, r["viatura"].replace("V",""), ha="center", va="center", color="#333333", fontsize=7, fontweight="bold", zorder=4)

    ax.set_aspect('equal')
    ax.set_xlim(-10, 130)
    ax.set_ylim(-30, 60)
    ax.set_xticks(np.arange(0, 130, 20))
    ax.set_yticks(np.arange(-20, 60, 20))
    ax.grid(True, linestyle=":", alpha=0.6)
    
    for spine in ['top', 'right', 'left', 'bottom']: 
        ax.spines[spine].set_visible(False)
        
    fig.tight_layout()
    return fig

def gerar_animacao_html5(dados, params, duracao_real_segundos):
    """
    Gera um componente HTML5 com gestão inteligente de lotação.
    Camioes que excedam a capacidade do parque (16) aguardam no exterior.
    """
    registos_js = json.dumps(dados["registos"])
    
    html_code = f"""
    <div style="width: 100%; display: flex; flex-direction: column; align-items: center; font-family: sans-serif;">
        <div id="relogio" style="font-size: 24px; font-weight: bold; color: #333; margin-bottom: 10px;">
            🕒 08:00h
        </div>
        <canvas id="simCanvas" width="900" height="400" style="background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);"></canvas>
    </div>

    <script>
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const relogio = document.getElementById('relogio');

        const registos = {registos_js};
        const numCais = {params['num_cais']};
        const duracaoSim = {params['duracao_sim']};
        const duracaoRealMs = {duracao_real_segundos} * 1000;
        
        const T_MOV = 1.5;

        // 1. Pré-Processamento: Gestor de Parqueamento
        let ocupacaoParque = new Array(16).fill(null);
        registos.forEach(r => {{
            let slot = -1;
            for(let i=0; i<16; i++) {{
                // Verifica se o lugar está livre (com uma margem para a manobra de saída)
                if (ocupacaoParque[i] === null || ocupacaoParque[i] <= r.t_chegada) {{
                    slot = i;
                    ocupacaoParque[i] = r.t_inicio_carga + 0.5; 
                    break;
                }}
            }}
            r.slot = slot;
            r.isExterior = (slot === -1);
        }});

        // Conversão Escala: X(-40 a 130) -> 900px (Espaço extra à esquerda para a Via Pública)
        function calcX(metrosX) {{ return ((metrosX + 40) / 170) * 900; }}
        function calcY(metrosY) {{ return 400 - (((metrosY + 30) / 90) * 400); }}

        const NOS = {{
            SPAWN: {{x: -40, y: 0}},           // Ponto de origem fora do ecrã
            PORT_IN: {{x: 0, y: 0}},
            CRUZ_FILA: {{x: 54, y: 0}},     
            SAIDA_FILA: {{x: 54, y: 48}},   
            CRUZ_TOPO_EIXO: {{x: 100, y: 48}}, 
            CRUZ_EIXO_IN: {{x: 100, y: 0}}, 
            CRUZ_EIXO_OUT: {{x: 100, y: -15}}, 
            PORT_OUT: {{x: 0, y: -15}}
        }};

        function desenharInfraestrutura(tAtual) {{
            // Estradas Gerais
            ctx.lineWidth = 25;
            ctx.strokeStyle = "#E0E0E0";
            ctx.lineCap = "round";
            ctx.lineJoin = "round";
            
            ctx.beginPath();
            ctx.moveTo(calcX(NOS.SPAWN.x), calcY(NOS.SPAWN.y)); // Começa na via exterior
            ctx.lineTo(calcX(NOS.CRUZ_EIXO_IN.x), calcY(NOS.CRUZ_EIXO_IN.y));
            ctx.lineTo(calcX(NOS.CRUZ_EIXO_OUT.x), calcY(NOS.CRUZ_EIXO_OUT.y));
            ctx.lineTo(calcX(NOS.PORT_OUT.x), calcY(NOS.PORT_OUT.y));
            ctx.stroke();

            // Corredor Central do Parque
            ctx.beginPath();
            ctx.moveTo(calcX(NOS.CRUZ_FILA.x), calcY(NOS.CRUZ_FILA.y));
            ctx.lineTo(calcX(NOS.SAIDA_FILA.x), calcY(NOS.SAIDA_FILA.y));
            ctx.lineTo(calcX(NOS.CRUZ_TOPO_EIXO.x), calcY(NOS.CRUZ_TOPO_EIXO.y));
            ctx.lineTo(calcX(NOS.CRUZ_EIXO_IN.x), calcY(NOS.CRUZ_EIXO_IN.y));
            ctx.stroke();

            // Espinhas de Navegação (Amarelo)
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "#F2C94C"; 
            
            ctx.beginPath();
            ctx.moveTo(calcX(NOS.CRUZ_FILA.x), calcY(NOS.CRUZ_FILA.y));
            ctx.lineTo(calcX(NOS.SAIDA_FILA.x), calcY(NOS.SAIDA_FILA.y));
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(calcX(NOS.CRUZ_TOPO_EIXO.x), calcY(NOS.CRUZ_TOPO_EIXO.y));
            ctx.lineTo(calcX(NOS.CRUZ_EIXO_OUT.x), calcY(NOS.CRUZ_EIXO_OUT.y));
            ctx.stroke();

            // Recinto do Parque e Loteamento (16 Lugares)
            ctx.fillStyle = "rgba(55, 138, 221, 0.15)";
            ctx.strokeStyle = "#378ADD";
            ctx.lineWidth = 2;
            let pX = calcX(28), pY = calcY(50), pW = calcX(80) - calcX(28), pH = calcY(10) - calcY(50);
            ctx.fillRect(pX, pY, pW, pH);
            ctx.strokeRect(pX, pY, pW, pH);
            ctx.fillStyle = "#333";
            ctx.font = "bold 11px Arial";
            ctx.textAlign = "center";
            ctx.fillText("PARQUE DE ESPERA (LOTAÇÃO: 16)", calcX(54), calcY(52));

            ctx.lineWidth = 1;
            ctx.strokeStyle = "rgba(55, 138, 221, 0.7)";
            ctx.fillStyle = "rgba(255, 255, 255, 0.5)";
            for (let s = 0; s < 16; s++) {{ 
                let col = s % 2;
                let row = Math.floor(s / 2);
                let cX_m = col === 0 ? 40 : 68; 
                let cY_m = 45 - (row * 4.5);
                let larg_m = 16.5; let alt_m = 2.5;
                
                let bx = calcX(cX_m - (larg_m/2)); let by = calcY(cY_m + (alt_m/2));
                let bw = calcX(cX_m + (larg_m/2)) - bx; let bh = calcY(cY_m - (alt_m/2)) - by;
                
                ctx.fillRect(bx, by, bw, bh); ctx.strokeRect(bx, by, bw, bh);
                
                ctx.beginPath(); ctx.strokeStyle = "#F2C94C";
                ctx.moveTo(calcX(54), calcY(cY_m)); ctx.lineTo(calcX(cX_m), calcY(cY_m)); ctx.stroke();
                
                ctx.fillStyle = "#378ADD"; ctx.font = "8px Arial"; ctx.textAlign = "left";
                ctx.fillText("P" + (s+1), bx + 2, by + 10);
                
                ctx.fillStyle = "rgba(255, 255, 255, 0.5)"; ctx.strokeStyle = "rgba(55, 138, 221, 0.7)";
            }}

            // Cais
            for(let i=0; i<numCais; i++) {{
                let yCais = 10 + (i * 8);
                let cX_m = NOS.CRUZ_EIXO_IN.x + 8.25; 
                
                ctx.fillStyle = "rgba(29, 158, 117, 0.8)";
                let cX = calcX(NOS.CRUZ_EIXO_IN.x), cY = calcY(yCais + 2), cW = calcX(120) - calcX(100), cH = calcY(yCais - 2) - calcY(yCais + 2);
                ctx.fillRect(cX, cY, cW, cH);
                ctx.fillStyle = "#333"; ctx.font = "bold 9px Arial";
                ctx.fillText("Cais " + (i+1), calcX(110), calcY(yCais - 3));

                ctx.beginPath(); ctx.strokeStyle = "#F2C94C";
                ctx.moveTo(calcX(NOS.CRUZ_EIXO_IN.x), calcY(yCais)); ctx.lineTo(calcX(cX_m), calcY(yCais)); ctx.stroke();
            }}
            
            // Portarias e Exterior
            ctx.fillStyle = "grey"; ctx.textAlign = "center";
            ctx.fillText("PORTARIA IN", calcX(0), calcY(5));
            ctx.fillText("PORTARIA OUT", calcX(0), calcY(-20));
            
            // UI DINÂMICA: Alerta de Camiões no Exterior
            let extCount = registos.filter(x => x.isExterior && tAtual >= x.t_chegada && tAtual < x.t_inicio_carga).length;
            if (extCount > 0) {{
                ctx.fillStyle = "rgba(216, 90, 48, 0.1)";
                ctx.fillRect(calcX(-38), calcY(12), calcX(-5) - calcX(-38), calcY(-5) - calcY(12));
                ctx.fillStyle = "#D85A30";
                ctx.font = "bold 10px Arial";
                ctx.fillText("⚠️ VIA PÚBLICA", calcX(-22), calcY(8));
                ctx.fillText(extCount + " Viatura(s) em Espera", calcX(-22), calcY(2));
            }}
        }}

        function lerp(start, end, amt) {{ return (1 - amt) * start + amt * end; }}

        function getPosOnPath(path, progress) {{
            if (path.length === 1) return path[0];
            let totalDist = 0; let segments = [];
            for(let i = 0; i < path.length - 1; i++) {{
                let dx = path[i+1].x - path[i].x; let dy = path[i+1].y - path[i].y;
                let len = Math.sqrt(dx*dx + dy*dy);
                segments.push({{ start: path[i], end: path[i+1], len: len }}); totalDist += len;
            }}

            let targetDist = progress * totalDist; let currentDist = 0;
            for(let i = 0; i < segments.length; i++) {{
                let seg = segments[i];
                if (currentDist + seg.len >= targetDist || i === segments.length - 1) {{
                    let segProg = (seg.len === 0) ? 1 : (targetDist - currentDist) / seg.len;
                    return {{ x: lerp(seg.start.x, seg.end.x, segProg), y: lerp(seg.start.y, seg.end.y, segProg) }};
                }}
                currentDist += seg.len;
            }}
            return path[path.length - 1];
        }}

        function desenharViaturas(tAtual) {{
            // Descobre os camiões ativos no exterior para formar uma fila ordenada
            let exteriorAtivos = registos.filter(x => x.isExterior && tAtual >= x.t_chegada && tAtual < x.t_inicio_carga)
                                         .sort((a,b) => a.t_chegada - b.t_chegada);

            registos.forEach(r => {{
                let tIn = r.t_chegada; let tLoad = r.t_inicio_carga; let tOut = r.t_fim_carga;

                if(tAtual < tIn || tAtual > tOut + T_MOV) return;

                let idNum = parseInt(r.viatura.replace("V", ""));
                let caisIdx = idNum % numCais; let yCais = 10 + (caisIdx * 8);
                let pCais = {{x: NOS.CRUZ_EIXO_IN.x + 8.25, y: yCais}};
                let pCaisEixo = {{x: NOS.CRUZ_EIXO_IN.x, y: yCais}}; 
                
                let pFila, pFilaEixo;
                if (!r.isExterior) {{
                    let col = r.slot % 2; let row = Math.floor(r.slot / 2);
                    pFila = {{x: col === 0 ? 40 : 68, y: 45 - (row * 4.5)}};
                    pFilaEixo = {{x: 54, y: pFila.y}};
                }} else {{
                    let extIdx = exteriorAtivos.findIndex(x => x.viatura === r.viatura);
                    if (extIdx === -1) extIdx = 0; 
                    pFila = {{x: -12 - (extIdx * 12), y: 0}}; // Fila horizontal na Via Pública
                    pFilaEixo = pFila; 
                }}
                
                let pos = null; let cor = "#D85A30"; 

                if (tLoad === tIn) {{ 
                    if (tAtual < tIn + T_MOV) {{
                        let prog = (tAtual - tIn) / T_MOV;
                        let rota = [NOS.SPAWN, NOS.PORT_IN, NOS.CRUZ_EIXO_IN, pCaisEixo, pCais];
                        pos = getPosOnPath(rota, prog);
                    }} else if (tAtual <= tOut) {{
                        pos = pCais; cor = "#1D9E75";
                    }} else {{
                        let prog = (tAtual - tOut) / T_MOV;
                        let rota = [pCais, pCaisEixo, NOS.CRUZ_EIXO_OUT, NOS.PORT_OUT];
                        pos = getPosOnPath(rota, prog);
                    }}
                }} else {{ 
                    if (tAtual < tIn + T_MOV) {{
                        let prog = (tAtual - tIn) / T_MOV;
                        let rota = r.isExterior ? [NOS.SPAWN, pFila] : [NOS.SPAWN, NOS.PORT_IN, NOS.CRUZ_FILA, pFilaEixo, pFila];
                        pos = getPosOnPath(rota, prog);
                    }} else if (tAtual <= tLoad) {{
                        pos = pFila;
                    }} else if (tAtual < tLoad + T_MOV) {{
                        let prog = (tAtual - tLoad) / T_MOV;
                        let rota = r.isExterior ? [pFila, NOS.PORT_IN, NOS.CRUZ_EIXO_IN, pCaisEixo, pCais] : [pFila, pFilaEixo, NOS.SAIDA_FILA, NOS.CRUZ_TOPO_EIXO, NOS.CRUZ_EIXO_IN, pCaisEixo, pCais];
                        pos = getPosOnPath(rota, prog);
                    }} else if (tAtual <= tOut) {{
                        pos = pCais; cor = "#1D9E75";
                    }} else {{
                        let prog = (tAtual - tOut) / T_MOV;
                        let rota = [pCais, pCaisEixo, NOS.CRUZ_EIXO_OUT, NOS.PORT_OUT];
                        pos = getPosOnPath(rota, prog);
                    }}
                }}

                if(pos !== null) {{
                    ctx.beginPath();
                    ctx.arc(calcX(pos.x), calcY(pos.y), 8, 0, 2 * Math.PI);
                    ctx.fillStyle = cor; ctx.fill();
                    ctx.lineWidth = 1; ctx.strokeStyle = "#fff"; ctx.stroke();
                    
                    ctx.fillStyle = "#fff"; ctx.font = "bold 9px Arial"; ctx.textAlign = "center";
                    ctx.fillText(idNum, calcX(pos.x), calcY(pos.y) + 3);
                }}
            }});
        }}

        let startTime = null;
        function animar(timestamp) {{
            if (!startTime) startTime = timestamp;
            let elapsedMs = timestamp - startTime;
            
            let tAtual = (elapsedMs / duracaoRealMs) * duracaoSim;
            if (tAtual > duracaoSim) tAtual = duracaoSim;
            
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            desenharInfraestrutura(tAtual);
            desenharViaturas(tAtual);
            
            let horas = 8 + Math.floor(tAtual / 60);
            let mins = Math.floor(tAtual % 60);
            relogio.innerText = "🕒 " + horas.toString().padStart(2, '0') + ":" + mins.toString().padStart(2, '0') + "h";

            if (elapsedMs < duracaoRealMs) {{
                window.requestAnimationFrame(animar);
            }} else {{
                relogio.innerText += " (TURNO CONCLUÍDO)";
            }}
        }}

        window.requestAnimationFrame(animar);
    </script>
    """
    return html_code


# ==========================================
# INTERFACE WEB STREAMLIT
# ==========================================

def main():
    st.set_page_config(page_title="Simulação Logística", layout="wide")
    st.title("Simulação Logística — Zona de Expedição")
    
    # --- PAINEL ESQUERDO (SIDEBAR) ---
# 1. Limites Estratégicos (Capacidade Instalada)
    st.sidebar.header("⚙️ Limites do Simulador")
    st.sidebar.caption("Defina os valores máximos permitidos para as barras de controlo.")
    col1, col2 = st.sidebar.columns(2)
    max_cais = col1.number_input("Max. Cais", min_value=1, max_value=50, value=5)
    max_intervalo = col2.number_input("Max. Intervalo", min_value=1, max_value=120, value=30)
    max_tempo_carga = st.sidebar.number_input("Max. Tempo de Carga", min_value=1, max_value=240, value=60)
    
    st.sidebar.divider()
    
    # 2. Configuração Operacional (Barras Deslizantes Dinâmicas)
    st.sidebar.header("📋 Operação do Turno")
    st.sidebar.caption("Ajuste o cenário diário dentro dos limites acima.")
    
    # O valor máximo (segundo argumento) agora lê as variáveis definidas em cima.
    # O 'value' (último argumento) usa a função min() para evitar erros se o utilizador baixar o limite abaixo do valor atual.
    num_cais = st.sidebar.slider("Nº de Cais a operar", 1, int(max_cais), min(2, int(max_cais)))
    intervalo = st.sidebar.slider("Intervalo Chegada (min)", 1, int(max_intervalo), min(8, int(max_intervalo)))
    tempo_carga = st.sidebar.slider("Tempo de Carga (min)", 1, int(max_tempo_carga), min(10, int(max_tempo_carga)))
    
    st.sidebar.divider()
    
    # 3. Controlo da Animação
    st.sidebar.header("▶️ Controlo da Animação")
    duracao_animacao = st.sidebar.slider("Duração da Animação (segundos)", 5, 120, 30)
    btn_iniciar = st.sidebar.button("▶ Iniciar Simulação Animada")
    
    params = {
        "nome": "Cenário Personalizado",
        "num_cais": num_cais,
        "intervalo_chegada": intervalo,
        "tempo_carga": tempo_carga,
        "duracao_sim": 480  # 8 Horas de operação
    }

    # Execução do motor base (cálculos instantâneos)
    dados = executar_simulacao(params)
    indicadores = calcular_indicadores(dados, params)

    # Exibição de KPIs
    st.markdown("### Indicadores Finais do Turno")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Viaturas Atendidas", f"{indicadores['total_viaturas']}")
    col2.metric("Espera Média", f"{indicadores['espera_media']:.1f} min")
    col3.metric("Taxa de Ocupação", f"{indicadores['taxa_ocupacao_pct']}%")
    col4.metric("Fila Máxima", f"{indicadores['tamanho_max_fila']} viat.")

    st.divider()

    # --- ZONA DE ANIMAÇÃO DINÂMICA ---
    st.markdown("### Mapa do Armazém em Tempo Real")
    
    # Criar um placeholder para os gráficos não se acumularem verticalmente
    placeholder_mapa = st.empty()
    placeholder_tempo = st.empty()

    if btn_iniciar:
        # Entrega o código HTML e JSON ao navegador
        codigo_html_animacao = gerar_animacao_html5(dados, params, duracao_animacao)
        
        # O Streamlit renderiza a janela isolada onde o JavaScript ganha vida
        components.html(codigo_html_animacao, height=450)
    else:
        st.info("Configure os parâmetros na esquerda e clique em 'Iniciar' para ver a movimentação das viaturas a 60 FPS.")

def grafico_chegadas_timeline(dados, params):
    """
    Gera uma linha temporal (timeline) dos momentos exatos de chegada de cada viatura.
    """
    registos = dados["registos"]
    if not registos:
        return None

    tempos_chegada = [r["t_chegada"] for r in registos]
    ids_viaturas = [r["viatura"] for r in registos]
    
    fig, ax = plt.subplots(figsize=(10, 2))
    
    # Desenhar linhas verticais no eixo do tempo (pings de chegada)
    ax.vlines(tempos_chegada, ymin=0, ymax=1, color="#D85A30", linewidth=1.5, alpha=0.8)
    ax.plot(tempos_chegada, np.ones(len(tempos_chegada)), "o", color="#D85A30", markersize=5)
    
    # Adicionar o ID da viatura (texto inclinado para leitura específica)
    for t, id_v in zip(tempos_chegada, ids_viaturas):
        ax.text(t, 1.15, id_v, rotation=45, ha='center', va='bottom', fontsize=7, color="#333333")
        
    ax.set_title("Radar de Portaria (Momentos Exatos de Chegada)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Tempo (min)")
    ax.set_yticks([]) # Ocultar eixo vertical por ser irrelevante
    ax.set_ylim(0, 2.5) # Margem superior para acomodar o texto inclinado
    ax.set_xlim(0, params["duracao_sim"])
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    
    # Remover bordas superiores e laterais para um aspeto limpo
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)
        
    fig.tight_layout()
    return fig

if __name__ == "__main__":
    main() # onde visualizamos a nossa solução a funcionar
