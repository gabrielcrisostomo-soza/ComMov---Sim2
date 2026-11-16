import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt, comb, factorial


def gray(index):
    return index ^ (index >> 1)


def gen_constellation(tipo, N):
    M = 2**N

    if tipo == "ask":
        pos = np.arange(M)
        simbolos = gray(pos)
        const = 2.0*pos - (M-1)

    elif tipo == "qam":
        
        lado = 2**(N//2)
        pos = np.arange(M)

        i = pos//lado
        q = pos % lado

        simbolos = (gray(i) << (N//2)) | gray(q)
        const = (
            (2.0*i - (lado-1))
            + 1j*(2.0*q - (lado-1))
        )

    
    const = const / np.sqrt(np.mean(np.abs(const)**2))

    return const, simbolos


def decisor(amostras_rx, const, simbolos, envoltoria):
    estimados = []
    s_estimados = []

    for k in range(len(amostras_rx)):
        distancias = np.abs(
            amostras_rx[k] - const*envoltoria[k]
        )**2

        indice = np.argmin(distancias)
        estimados.append(const[indice])
        s_estimados.append(simbolos[indice])

    return np.array(estimados), np.array(s_estimados)


def gen_message(Nt, M, rng, constelacao, simbolos):
    uniform = rng.integers(0, M, size=Nt)
    amostras_tx = constelacao[uniform]
    simbolos_tx = simbolos[uniform]

    return amostras_tx, simbolos_tx


def rayleigh_diverso(Nt, L, rng):
    x = rng.normal(0, 1, size=(Nt, L))
    y = rng.normal(0, 1, size=(Nt, L))

    return np.sqrt((x**2 + y**2)/2)


def combinar_sc(amostras_rx, envoltorias):
    recebidos = []
    ganhos = []

    for k in range(len(amostras_rx)):
        ramo = np.argmax(envoltorias[k])

        recebidos.append(amostras_rx[k, ramo])
        ganhos.append(envoltorias[k, ramo])

    return np.asarray(recebidos), np.asarray(ganhos)


def combinar_mrc(amostras_rx, envoltorias):
    recebidos = []
    ganhos = []

    for k in range(len(amostras_rx)):
        recebido = np.sum(envoltorias[k] * amostras_rx[k])
        ganho = np.sum(envoltorias[k]**2)

        recebidos.append(recebido)
        ganhos.append(ganho)

    return np.asarray(recebidos), np.asarray(ganhos)


def combinar_egc(amostras_rx, envoltorias):
    recebidos = []
    ganhos = []

    for k in range(len(amostras_rx)):
        recebido = np.sum(amostras_rx[k])
        ganho = np.sum(envoltorias[k])

        recebidos.append(recebido)
        ganhos.append(ganho)

    return np.asarray(recebidos), np.asarray(ganhos)


def sep_sc(L, M, snr_media):
    am = 2*(M-1)/M #ask
    bm = 6/(M**2-1)

    soma = 0
    for k in range(L):
        a = comb(L-1, k)
        b = (L/(2*(k+1)))*(-1)**k
        x = bm * snr_media / (2 * (k + 1))
        c = 1 - np.sqrt(x / (1 + x))
        soma += a*b*c

    return am*soma

def sep_mrc(L, M, snr_media):
    am = 2*(M-1)/M #ask
    bm = 6/(M**2-1)
    mi = np.sqrt( (bm*snr_media/2)/(1+bm*snr_media/2) )

    soma = 0
    for k in range(L):
        soma += comb(L-1+k,k)*( (1+mi)/2 )**k

    return (am*( (1-mi)/2 )**L)*soma


def main():
    # parametros
    rng = np.random.default_rng(2)
    Nt = 10**6
    L_valores = [1, 2, 3, 4]

    inicio, fim = -5, 30
    snr_db = np.arange(inicio, fim+1, 5)
    snr_linear = 10**(snr_db/10)

    snr_db_teoria = np.linspace(inicio, fim, 200)
    snr_linear_teoria = 10.0**(snr_db_teoria / 10)

    N = 2  # M = 4
    tipo = "ask"

    constelacao, simbolos = gen_constellation(tipo, N)

    amostras_tx, simbolos_tx = gen_message(Nt, 2**N, rng, constelacao, simbolos)

    resultados = {
        L: {
            tecnica: {
                "ser": []
            }
            for tecnica in ["sc", "mrc", "egc"]
        }
        for L in L_valores
    }

    teorico = {
        L: {
            tecnica: {
                "sep": []
            }
            for tecnica in ["sc", "mrc"]
        }
        for L in L_valores
    }

    envoltorias_ray = rayleigh_diverso(
        Nt, max(L_valores), rng
    )

    # Simulação
    for L in L_valores:
        envoltorias_L = envoltorias_ray[:, :L]

        for snr_media in snr_linear:
            var_ruido = 1/(2*snr_media)

            ruido = np.sqrt(var_ruido) * (rng.standard_normal((Nt, L)))

            amostras_rx = (envoltorias_L * amostras_tx[:, None] + ruido)

            #   ====    ====    ====     ====    ====    ====    ====
            rx_sc, ganho_sc     = combinar_sc(amostras_rx, envoltorias_L)
            rx_mrc, ganho_mrc   = combinar_mrc(amostras_rx, envoltorias_L)
            rx_egc, ganho_egc   = combinar_egc(amostras_rx, envoltorias_L)
            
            
            _, simbolos_rx_sc = decisor(rx_sc, constelacao, simbolos, ganho_sc)
            _, simbolos_rx_mrc= decisor(rx_mrc, constelacao, simbolos, ganho_mrc)
            _, simbolos_rx_egc= decisor(rx_egc, constelacao, simbolos, ganho_egc)

            erros_sc = np.count_nonzero(simbolos_tx != simbolos_rx_sc)
            erros_mrc = np.count_nonzero(simbolos_tx != simbolos_rx_mrc)
            erros_egc = np.count_nonzero(simbolos_tx != simbolos_rx_egc)

            ser_sc = erros_sc/Nt
            ser_mrc = erros_mrc/Nt
            ser_egc = erros_egc/Nt


            resultados[L]["sc"]["ser"].append(ser_sc)
            resultados[L]["mrc"]["ser"].append(ser_mrc)
            resultados[L]["egc"]["ser"].append(ser_egc)

        for snr_media in snr_linear_teoria:
            teorico[L]["sc"]["sep"].append(sep_sc(L, 2**N, snr_media))
            teorico[L]["mrc"]["sep"].append(sep_mrc(L,2**N, snr_media))

    # ========================================

    cores = {
        "sc": "#0050D8",   
        "mrc": "#E00020",  
        "egc": "#008040"   
    }

    marcadores = {"sc": "o", "mrc": "s", "egc": "^"}

    # ========================================

    fig, axes = plt.subplots(
        1, len(L_valores),
        figsize=(5 * len(L_valores), 5),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for ax, L in zip(axes[0], L_valores):

        
        for tecnica in ["sc", "mrc", "egc"]:
            ser = np.asarray(resultados[L][tecnica]["ser"])

            
            ser_plot = np.where(ser > 0, ser, np.nan)

            ax.semilogy(
                snr_db,
                ser_plot,
                marker=marcadores[tecnica],
                markersize=4.5,
                markerfacecolor=cores[tecnica],
                markeredgecolor="black",
                markeredgewidth=0.5,
                linestyle="--" if tecnica == "egc" else "none",
                linewidth=1.5,
                color=cores[tecnica],
                label=f"{tecnica.upper()} — SER simulada",
                zorder=3
            )

        # Teoria
        for tecnica in ["sc", "mrc"]:
            ax.semilogy(
                snr_db_teoria,
                teorico[L][tecnica]["sep"],
                linestyle="-",
                linewidth=1.5,
                color=cores[tecnica],
                label=f"{tecnica.upper()} — SEP teórica",
                zorder=2
            )

        ax.set_title(f"L = {L}")
        ax.set_xlabel("SNR média por símbolo por ramo (dB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Probabilidade de erro de símbolo")
    axes[0, -1].legend(fontsize=8)

    fig.suptitle(f"{2**N}-{tipo.upper()} em Rayleigh")
    fig.tight_layout()

    snr_analise_db = 10
    indice_ser = np.flatnonzero(snr_db == snr_analise_db)[0]
    snr_analise_linear = 10.0**(snr_analise_db / 10)

    fig_ser_L, ax_ser_L = plt.subplots(figsize=(8, 5))

    # SER simulada para cada número de ramos
    for tecnica in ["sc", "mrc", "egc"]:
        ser_por_L = np.asarray([
            resultados[L][tecnica]["ser"][indice_ser]
            for L in L_valores
        ])

        ser_plot = np.where(ser_por_L > 0, ser_por_L, np.nan)

        ax_ser_L.semilogy(
            L_valores,
            ser_plot,
            marker=marcadores[tecnica],
            markersize=4.5,
            markerfacecolor=cores[tecnica],
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="--" if tecnica == "egc" else "none",
            linewidth=1.5,
            color=cores[tecnica],
            label=f"{tecnica.upper()} — SER simulada",
            zorder=3
        )

    # SEP teórica nos mesmos valores inteiros de L
    sep_por_L = {
        "sc": [
            sep_sc(L, 2**N, snr_analise_linear)
            for L in L_valores
        ],
        "mrc": [
            sep_mrc(L, 2**N, snr_analise_linear)
            for L in L_valores
        ]
    }

    for tecnica in ["sc", "mrc"]:
        ax_ser_L.semilogy(
            L_valores,
            sep_por_L[tecnica],
            linestyle="-",
            linewidth=1.5,
            color=cores[tecnica],
            label=f"{tecnica.upper()} — SEP teórica",
            zorder=2
        )

    ax_ser_L.set_xticks(L_valores)
    ax_ser_L.set_xlabel("Número de ramos L")
    ax_ser_L.set_ylabel("Probabilidade de erro de símbolo")
    ax_ser_L.set_title(
        f"{2**N}-{tipo.upper()} em Rayleigh\n"
        f"SNR média por símbolo por ramo = {snr_analise_db} dB"
    )
    ax_ser_L.grid(True, which="both", alpha=0.3)
    ax_ser_L.set_axisbelow(True)
    ax_ser_L.legend(fontsize=8)

    fig_ser_L.tight_layout()


    plt.show()
            
            





if __name__ == "__main__":
    main()