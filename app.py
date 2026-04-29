import streamlit
import simpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np 


CENARIOS = [
    {
        "nome":              "Cenário Base",
        "num_cais":          2,
        "intervalo_chegada": 8,    # minutos entre chegadas de viaturas
        "tempo_carga":       10,   # minutos por operação de carga
        "duracao_sim":       120,  # minutos totais de simulação
    },
    {
        "nome":              "Congestionamento",
        "num_cais":          1,
        "intervalo_chegada": 5,
        "tempo_carga":       15,
        "duracao_sim":       120,
    },
    {
        "nome":              "Capacidade Ampliada",
        "num_cais":          3,
        "intervalo_chegada": 8,
        "tempo_carga":       10,
        "duracao_sim":       240,
    },
]

def processo_viatura(env, id_viatura, cais, params, dados):
    """
    Processo SimPy que representa o ciclo de vida de uma viatura:
    chegada → espera na fila → operação de carga → saída.
    """
    t_chegada = env.now

    # Regista o comprimento da fila ANTES de fazer o pedido
    dados["historico_fila"].append((env.now, len(cais.queue)))

    # Pede um cais (aguarda se não houver disponível)
    with cais.request() as pedido:
        yield pedido  # <-- aqui a viatura fica em fila se necessário

        t_inicio_carga = env.now
        espera = t_inicio_carga - t_chegada

        # Regista ocupação: +1 cais em uso
        dados["historico_ocupacao"].append((env.now, cais.count))

        # Simula o tempo de carregamento
        yield env.timeout(params["tempo_carga"])

        t_fim_carga = env.now

        # Regista que o cais ficou livre
        dados["historico_ocupacao"].append((env.now, cais.count - 1))
        dados["historico_fila"].append((env.now, len(cais.queue)))

        # Guarda registo desta viatura
        dados["registos"].append({
            "viatura":        id_viatura,
            "t_chegada":      t_chegada,
            "t_inicio_carga": t_inicio_carga,
            "t_fim_carga":    t_fim_carga,
            "espera":         espera,
            "tempo_carga":    params["tempo_carga"],
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
        "registos":           [],   # lista de dicts por viatura
        "historico_fila":     [],   # (tempo, comprimento_fila)
        "historico_ocupacao": [],   # (tempo, num_cais_ocupados)
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
        return {}

    esperas = [r["espera"] for r in registos]

    # Taxa de ocupação média dos cais
    if dados["historico_ocupacao"]:
        tempos_oc = [d[0] for d in dados["historico_ocupacao"]]
        valores_oc = [d[1] for d in dados["historico_ocupacao"]]
        # Média ponderada pelo tempo
        area = np.trapz(valores_oc, tempos_oc)
        taxa_ocup = (area / params["duracao_sim"]) / params["num_cais"] * 100
    else:
        taxa_ocup = 0

    # Tamanho máximo da fila
    if dados["historico_fila"]:
        max_fila = max(d[1] for d in dados["historico_fila"])
    else:
        max_fila = 0

    return {
        "total_viaturas":     len(registos),
        "espera_media":       np.mean(esperas),
        "espera_max":         np.max(esperas),
        "espera_min":         np.min(esperas),
        "taxa_ocupacao_pct":  round(taxa_ocup, 1),
        "tamanho_max_fila":   max_fila,
    }


def imprimir_relatorio(params, indicadores):
    print(f"\n{'='*55}")
    print(f"  {params['nome']}")
    print(f"{'='*55}")
    print(f"  Nº de cais:          {params['num_cais']}")
    print(f"  Intervalo chegada:   {params['intervalo_chegada']} min")
    print(f"  Tempo de carga:      {params['tempo_carga']} min")
    print(f"  Duração simulação:   {params['duracao_sim']} min")
    print(f"  {'─'*45}")
    print(f"  Viaturas atendidas:  {indicadores['total_viaturas']}")
    print(f"  Espera média:        {indicadores['espera_media']:.1f} min")
    print(f"  Espera máxima:       {indicadores['espera_max']:.0f} min")
    print(f"  Taxa de ocupação:    {indicadores['taxa_ocupacao_pct']}%")
    print(f"  Fila máxima:         {indicadores['tamanho_max_fila']} viaturas")
    print(f"{'='*55}") def grafico_fila_e_ocupacao(dados, params, ax_fila, ax_ocup):
    """
    Plota a evolução da fila e da ocupação dos cais num par de eixos.
    """
    # --- Fila ---
    if dados["historico_fila"]:
        tf = [d[0] for d in dados["historico_fila"]]
        vf = [d[1] for d in dados["historico_fila"]]
        ax_fila.step(tf, vf, where="post", color="#378ADD", linewidth=1.5)
        ax_fila.fill_between(tf, vf, step="post", alpha=0.15, color="#378ADD")
    ax_fila.set_ylabel("Nº viaturas em fila")
    ax_fila.set_title(params["nome"], fontsize=11, fontweight="bold")
    ax_fila.set_ylim(bottom=0)
    ax_fila.grid(True, alpha=0.3)

    # --- Ocupação ---
    if dados["historico_ocupacao"]:
        to = [d[0] for d in dados["historico_ocupacao"]]
        vo = [d[1] for d in dados["historico_ocupacao"]]
        # Percentagem de ocupação
        vo_pct = [v / params["num_cais"] * 100 for v in vo]
        ax_ocup.step(to, vo_pct, where="post", color="#1D9E75", linewidth=1.5)
        ax_ocup.fill_between(to, vo_pct, step="post", alpha=0.15, color="#1D9E75")
    ax_ocup.set_ylabel("Ocupação dos cais (%)")
    ax_ocup.set_xlabel("Tempo (min)")
    ax_ocup.set_ylim(0, 110)
    ax_ocup.axhline(100, color="red", linewidth=0.8, linestyle="--", alpha=0.5)
    ax_ocup.grid(True, alpha=0.3)

def grafico_comparativo(resultados):
    """
    Gráfico de barras comparando os cenários nos indicadores principais.
    """
    nomes = [r["params"]["nome"] for r in resultados]
    esperas = [r["indicadores"]["espera_media"] for r in resultados]
    ocup = [r["indicadores"]["taxa_ocupacao_pct"] for r in resultados]
    filas = [r["indicadores"]["tamanho_max_fila"] for r in resultados]

    x = np.arange(len(nomes))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Comparação entre Cenários", fontsize=13, fontweight="bold")

    cores = ["#378ADD", "#D85A30", "#1D9E75"]

    # Espera média
    axes[0].bar(x, esperas, color=cores, edgecolor="white", width=0.5)
    axes[0].set_title("Espera média (min)")
    axes[0].set_xticks(x); axes[0].set_xticklabels(nomes, fontsize=8)
    axes[0].set_ylim(bottom=0); axes[0].grid(axis="y", alpha=0.3)

    # Taxa de ocupação
    axes[1].bar(x, ocup, color=cores, edgecolor="white", width=0.5)
    axes[1].set_title("Taxa de ocupação (%)")
    axes[1].set_xticks(x); axes[1].set_xticklabels(nomes, fontsize=8)
    axes[1].set_ylim(0, 110); axes[1].axhline(100, color="red", linestyle="--", lw=0.8, alpha=0.5)
    axes[1].grid(axis="y", alpha=0.3)

    # Fila máxima
    axes[2].bar(x, filas, color=cores, edgecolor="white", width=0.5)
    axes[2].set_title("Fila máxima (viaturas)")
    axes[2].set_xticks(x); axes[2].set_xticklabels(nomes, fontsize=8)
    axes[2].set_ylim(bottom=0); axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparacao_cenarios.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  >> Gráfico comparativo guardado: comparacao_cenarios.png") 


def main():
    print("\n" + "="*55)
    print("  SIMULAÇÃO LOGÍSTICA — ZONA DE EXPEDIÇÃO")
    print("  Python + SimPy")
    print("="*55)

    resultados = []

    # --- Gráficos por cenário ---
    fig, axes = plt.subplots(
        len(CENARIOS), 2,
        figsize=(12, 4 * len(CENARIOS)),
        sharex=False
    )
    fig.suptitle("Evolução da Fila e Ocupação dos Cais por Cenário",
                 fontsize=13, fontweight="bold")

    for idx, params in enumerate(CENARIOS):
        dados = executar_simulacao(params)
        indicadores = calcular_indicadores(dados, params)
        imprimir_relatorio(params, indicadores)

        ax_fila = axes[idx][0]
        ax_ocup = axes[idx][1]
        grafico_fila_e_ocupacao(dados, params, ax_fila, ax_ocup)

        resultados.append({"params": params, "dados": dados, "indicadores": indicadores})

    plt.tight_layout()
    plt.savefig("evolucao_fila_ocupacao.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\n  >> Gráfico de evolução guardado: evolucao_fila_ocupacao.png")

    # --- Gráfico comparativo ---
    grafico_comparativo(resultados)

    print("\n  Simulação concluída. Verifica os gráficos gerados.")


if __name__ == "__main__":
    main() Este será o motor base. Gostaria de aplicar este script no github e gerar uma pagina web onde visualizamos a nossa solução a funcionar
