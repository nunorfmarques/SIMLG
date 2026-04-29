import simpy
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

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
    Gera chegadas de viaturas a intervalos regulares.
    """
    i = 0
    while True:
        yield env.timeout(params["intervalo_chegada"])
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
        area = np.trapz(valores_oc, tempos_oc)
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

# ==========================================
# INTERFACE WEB STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="Simulação Logística", layout="wide")
    st.title("Simulação Logística — Zona de Expedição")
    
    st.sidebar.header("Configuração do Cenário")
    num_cais = st.sidebar.slider("Nº de Cais", 1, 5, 2)
    intervalo = st.sidebar.slider("Intervalo Chegada (min)", 1, 30, 8)
    tempo_carga = st.sidebar.slider("Tempo de Carga (min)", 5, 60, 10)
    
    params = {
        "nome": "Cenário Personalizado",
        "num_cais": num_cais,
        "intervalo_chegada": intervalo,
        "tempo_carga": tempo_carga,
        "duracao_sim": 120
    }

    # Motor de Execução
    dados = executar_simulacao(params)
    indicadores = calcular_indicadores(dados, params)

    # Bloco de KPIs Analíticos
    st.markdown("### Indicadores de Desempenho")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Viaturas Atendidas", f"{indicadores['total_viaturas']}")
    col2.metric("Espera Média", f"{indicadores['espera_media']:.1f} min")
    col3.metric("Taxa de Ocupação", f"{indicadores['taxa_ocupacao_pct']}%")
    col4.metric("Fila Máxima", f"{indicadores['tamanho_max_fila']} viat.")

    st.divider()

    # Bloco de Visualização Gráfica
    st.markdown("### Monitorização do Armazém")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4)
    
    ax_fila = axes[0]
    ax_ocup = axes[1]
    
    grafico_fila_e_ocupacao(dados, params, ax_fila, ax_ocup)
    st.pyplot(fig)

if __name__ == "__main__":
    main() # onde visualizamos a nossa solução a funcionar
