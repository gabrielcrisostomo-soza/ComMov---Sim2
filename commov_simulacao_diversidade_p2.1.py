import numpy as np
import matplotlib.pyplot as plt
from math import erfc, sqrt, comb, factorial
from scipy.special import exp1, expn, gammainc


def rayleigh_diverso(Nt, L, rng):

    x = rng.normal(0, 1, size=(Nt, L))
    y = rng.normal(0, 1, size=(Nt, L))

    return np.sqrt((x**2 + y**2)/2)


def rice_diverso(Nt, L, K, rng):
    mx = np.sqrt(2 * K)

    x = rng.normal(mx, 1, size=(Nt, L))
    y = rng.normal(0, 1, size=(Nt, L))

    return np.sqrt((x**2 + y**2) / (2 * (K + 1)))

def snr_sc(snr_ramos):
    
    selection_combining = []

    for k in range(len(snr_ramos)):
        snr_max = np.max(snr_ramos[k])
        selection_combining.append(snr_max)

    return np.asarray(selection_combining)


def snr_mrc(snr_ramos):

    maximum_ratio_combining = []

    for k in range(len(snr_ramos)):
        snr_sum = np.sum(snr_ramos[k])
        maximum_ratio_combining.append(snr_sum)

    return np.asarray(maximum_ratio_combining)

def snr_egc(snr_ramos):

    equal_gain_combining = []
    L = len(snr_ramos[0])

    for k in range(len(snr_ramos)):
        egc_sum = np.sum(np.sqrt(snr_ramos[k]))
        egc_r = (1/L)*egc_sum**2
        equal_gain_combining.append(egc_r)

    return np.asarray(equal_gain_combining)
    

def capacidade_ergodica(snr_inst):
    return np.mean(np.log2(1+snr_inst))

def outage_simulada(snr, limiar):
    return np.mean(snr < limiar)

def capacidade_sc_teorica(L, snr_media):

    soma = 0
    for k in range(L):
        a = comb(L-1, k)
        b = np.exp((k+1)/snr_media)*((-1)**k)/(k+1)
        c = exp1((k+1)/snr_media)
        soma += a*b*c

    return L/np.log(2)*soma

def capacidade_mrc_teoria(L, snr_media):

    soma = 0
    for k in range(L):
        soma += expn(L-k, 1/snr_media)

    return (np.exp(1/snr_media)/np.log(2))*soma


#def op_sc_teoria(L, limiar, snr_media):
#    return ((1-np.exp(-limiar/snr_media))**L)

def op_mrc_teoria(L, limiar, snr_media):
    soma=0
    for k in range(L):
        soma+= (1/factorial(k))*(limiar/snr_media)**k

    return 1 - np.exp(-limiar/snr_media)*soma

def op_sc_teoria(L, limiar, snr_media):
    return (-np.expm1(-limiar / snr_media))**L

def main():


    #   ====    ====    parametros      ====    ====    
    rng = np.random.default_rng(1)
    Nt          = 10**5
    L_valores   = [1, 2, 4, 8]
    
    inicio, fim = -5, 30
    snr_db      = np.arange(inicio, fim+1, 5)
    snr_linear  = 10**(snr_db/10)

    snr_db_teoria       = np.linspace(inicio, fim, 200)
    snr_linear_teoria   = 10**(snr_db_teoria/10)


    snr_op_db           = 10
    snr_op_linear       = 10**(snr_op_db/10)
    limiares_db         = np.arange(-10,21,3)
    limiares_linear     = 10**(limiares_db/10)

    limiares_db_teoria = np.linspace(-10, 20, 200)
    limiares_linear_teoria = 10**(limiares_db_teoria/10)

    #   ====    ====    ====    ====    ====    ====
    

    resultados = {
        L:          {
            tecnica:    {
                "capacidade":[],
                "ser"       :[],
                "op"        :[],
            } for tecnica in ["sc", "mrc", "egc"]         
        }for L in L_valores
    }

    teorico =   {
        L:          {
            tecnica:    {
                "capacidade":[],
                "sep"       :[],
                "op"        :[]
            }for tecnica in ["sc", "mrc"]
        } for L in L_valores
    }
    


    #
    envoltorias_ray     = rayleigh_diverso(Nt, max(L_valores), rng)
    #snr_env_ray         = snr_linear*envoltorias_ray**2




    #   ====    ====    experimental====    ====    ====

    for L in L_valores:

        envoltorias_L = envoltorias_ray[:,:L]

        #   ====    ====    ====    capacidade  ====    ====    ====
        for snr_media in snr_linear:

            #               snr
            snr_inst = snr_media*envoltorias_L**2

            gamma_sc    = snr_sc(snr_inst)
            gamma_mrc   = snr_mrc(snr_inst)
            gamma_egc   = snr_egc(snr_inst)

            #               capacidade

            resultados[L]["sc"]["capacidade"].append(  capacidade_ergodica(gamma_sc  ))
            resultados[L]["mrc"]["capacidade"].append( capacidade_ergodica(gamma_mrc ))
            resultados[L]["egc"]["capacidade"].append( capacidade_ergodica(gamma_egc ))

            #teorico["sc"]["capacidade"].append(capacidade_sc_teorica(L, snr_media))
        

        #   ====    ====    ====    outage      ====    ====    ====
        snr_op = snr_op_linear*envoltorias_L**2
        gamma_sc_op = snr_sc(snr_op)
        gamma_mrc_op = snr_mrc(snr_op)
        gamma_egc_op = snr_egc(snr_op)

        for limiar in limiares_linear:

            resultados[L]["sc"]["op"].append(outage_simulada(gamma_sc_op, limiar))
            resultados[L]["mrc"]["op"].append(outage_simulada(gamma_mrc_op, limiar))
            resultados[L]["egc"]["op"].append(outage_simulada(gamma_egc_op, limiar))




    #   ====    ====    teorico     ====    ====    ====

        for gamma in snr_linear_teoria:
            teorico[L]["sc"]["capacidade"].append(capacidade_sc_teorica(L,gamma))
            teorico[L]["mrc"]["capacidade"].append(capacidade_mrc_teoria(L,gamma))

        for limiar_t in limiares_linear_teoria:
            teorico[L]["sc"]["op"].append(op_sc_teoria(L, limiar_t, snr_op_linear))
            teorico[L]["mrc"]["op"].append(op_mrc_teoria(L, limiar_t, snr_op_linear))


    indice = np.flatnonzero(snr_db == 10)[0]
    #rice

    K_db = 5
    K = 10.0**(K_db / 10)

    # Gerador separado para preservar os resultados atuais de Rayleigh
    rng_rice = np.random.default_rng(2)

    envoltorias_rice = rice_diverso(Nt, max(L_valores), K, rng_rice)

    capacidades_rice = {
        L: {
            tecnica: []
            for tecnica in ["sc", "mrc", "egc"]
        }
        for L in L_valores
    }

    for L in L_valores:
        envoltorias_L_rice = envoltorias_rice[:, :L]

        for snr_media in snr_linear:
            snr_ramos_rice = snr_media * envoltorias_L_rice**2

            gamma_sc = snr_sc(snr_ramos_rice)
            gamma_mrc = snr_mrc(snr_ramos_rice)
            gamma_egc = snr_egc(snr_ramos_rice)

            capacidades_rice[L]["sc"].append(capacidade_ergodica(gamma_sc))
            capacidades_rice[L]["mrc"].append(capacidade_ergodica(gamma_mrc))
            capacidades_rice[L]["egc"].append(capacidade_ergodica(gamma_egc))

    referencia_rice = capacidades_rice[1]["sc"][indice]

    ganhos_capacidade_rice = {
        tecnica: [
            capacidades_rice[L][tecnica][indice] - referencia_rice
            for L in L_valores
        ]
        for tecnica in ["sc", "mrc", "egc"]
    }

    #teste maquinado
    

    referencia = resultados[1]["sc"]["capacidade"][indice]

    ganhos_capacidade = {
        tecnica: [
            resultados[L][tecnica]["capacidade"][indice] - referencia
            for L in L_valores
        ]
        for tecnica in ["sc", "mrc", "egc"]
    }    

    limiar_analise_db = 5

    indice_op = np.flatnonzero(limiares_db == limiar_analise_db)[0]

    referencia_op = resultados[1]["sc"]["op"][indice_op]

    fatores_outage = {}

    for tecnica in ["sc", "mrc", "egc"]:
        probabilidades = np.asarray([
            resultados[L][tecnica]["op"][indice_op]
            for L in L_valores
        ])

        
        fatores = np.full(probabilidades.shape, np.nan)

        np.divide(
            referencia_op,
            probabilidades,
            out=fatores,
            where=probabilidades > 0
        )

        fatores_outage[tecnica] = fatores    
    
    # ========================================

    cores = {
        "sc": "#0050D8",   
        "mrc": "#E00020",  
        "egc": "#008040"  
    }

    marcadores = {"sc": "o", "mrc": "s", "egc": "^"}

    estilo_pontos = {
        "markersize": 4.5,
        "markeredgecolor": "black",
        "markeredgewidth": 0.5,
        "linewidth": 1.5,
        "zorder": 3
    }

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
            ax.plot(
                snr_db,
                resultados[L][tecnica]["capacidade"],
                marker=marcadores[tecnica],
                linestyle="--" if tecnica == "egc" else "none",
                color=cores[tecnica],
                markerfacecolor=cores[tecnica],
                label=f"{tecnica.upper()} — Simulada",
                **estilo_pontos
            )

        for tecnica in ["sc", "mrc"]:
            ax.plot(
                snr_db_teoria,
                teorico[L][tecnica]["capacidade"],
                linestyle="-",
                linewidth=1.5,
                color=cores[tecnica],
                label=f"{tecnica.upper()} — teórica",
                zorder=2
            )

        ax.set_title(f"Rayleigh — L = {L}")
        ax.set_xlabel("SNR média por ramo (dB)")
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0, 0].set_ylabel("Capacidade ergódica (bit/s/Hz)")
    axes[0, -1].legend(fontsize=8)

    fig.tight_layout()

    # ========================================

    fig_op, axes_op = plt.subplots(
        1, len(L_valores),
        figsize=(5 * len(L_valores), 5),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for ax, L in zip(axes_op[0], L_valores):

        for tecnica in ["sc", "mrc", "egc"]:
            op = np.asarray(resultados[L][tecnica]["op"])
            op_plot = np.where(op > 0, op, np.nan)

            ax.semilogy(
                limiares_db,
                op_plot,
                marker=marcadores[tecnica],
                linestyle="--" if tecnica == "egc" else "none",
                color=cores[tecnica],
                markerfacecolor=cores[tecnica],
                label=f"{tecnica.upper()} — Simulada",
                **estilo_pontos
            )

        for tecnica in ["sc", "mrc"]:
            ax.semilogy(
                limiares_db_teoria,
                teorico[L][tecnica]["op"],
                linestyle="-",
                linewidth=1.5,
                color=cores[tecnica],
                label=f"{tecnica.upper()} — teórica",
                zorder=2
            )

        
        ax.set_title(f"L = {L}")
        ax.set_xlabel("Limiar de SNR (dB)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_axisbelow(True)

    axes_op[0, 0].set_ylabel("Probabilidade de outage")
    axes_op[0, -1].legend(fontsize=8)

    fig_op.suptitle(
        f"Rayleigh — SNR média por ramo = {snr_op_db} dB"
    )
    fig_op.tight_layout()

    # ========================================

    fig_ganho, ax_ganho = plt.subplots(figsize=(8, 5))

    for tecnica in ["sc", "mrc", "egc"]:
        ax_ganho.plot(
            L_valores,
            ganhos_capacidade[tecnica],
            marker=marcadores[tecnica],
            linestyle="--" if tecnica == "egc" else "-",
            color=cores[tecnica],
            markerfacecolor=cores[tecnica],
            label=tecnica.upper(),
            **estilo_pontos
        )

    ax_ganho.set_xticks(L_valores)
    ax_ganho.set_xlabel("Número de ramos L")
    ax_ganho.set_ylabel("Ganho de capacidade sobre L = 1 (bit/s/Hz)")
    ax_ganho.set_title(
        f"Rayleigh — SNR média por ramo = {snr_db[indice]} dB"
    )
    ax_ganho.grid(True, alpha=0.3)
    ax_ganho.set_axisbelow(True)
    ax_ganho.legend()

    fig_ganho.tight_layout()

    # ========================================

    fig_ganho_op, ax_ganho_op = plt.subplots(figsize=(8, 5))

    for tecnica in ["sc", "mrc", "egc"]:
        ax_ganho_op.semilogy(
            L_valores,
            fatores_outage[tecnica],
            marker=marcadores[tecnica],
            linestyle="--" if tecnica == "egc" else "-",
            color=cores[tecnica],
            markerfacecolor=cores[tecnica],
            label=tecnica.upper(),
            **estilo_pontos
        )

    ax_ganho_op.set_xticks(L_valores)
    ax_ganho_op.set_xlabel("Número de ramos L")
    ax_ganho_op.set_ylabel(
        "Fator de redução da outage em relação a L = 1"
    )
    ax_ganho_op.set_title(
        f"Rayleigh — SNR média por ramo = {snr_op_db} dB\n"
        f"Limiar = {limiar_analise_db} dB"
    )
    ax_ganho_op.grid(True, which="both", alpha=0.3)
    ax_ganho_op.set_axisbelow(True)
    ax_ganho_op.legend()

    fig_ganho_op.tight_layout()

    # ========================================

    fig_comp, ax_comp = plt.subplots(figsize=(9, 6))

    for tecnica in ["sc", "mrc", "egc"]:

        
        ax_comp.plot(
            L_valores,
            ganhos_capacidade[tecnica],
            marker=marcadores[tecnica],
            linestyle="--" if tecnica == "egc" else "-",
            color=cores[tecnica],
            markerfacecolor=cores[tecnica],
            label=f"{tecnica.upper()} — Rayleigh",
            **estilo_pontos
        )

        ax_comp.plot(
            L_valores,
            ganhos_capacidade_rice[tecnica],
            marker=marcadores[tecnica],
            linestyle="--",
            color=cores[tecnica],
            markerfacecolor="white",
            label=f"{tecnica.upper()} — Rice",
            **estilo_pontos
        )

    ax_comp.set_xticks(L_valores)
    ax_comp.set_xlabel("Número de ramos L")
    ax_comp.set_ylabel("Ganho de capacidade sobre L = 1 (bit/s/Hz)")
    ax_comp.set_title(
        f"SNR média por ramo = {snr_db[indice]} dB\n"
        f"Rayleigh × Rice com K = {K_db} dB"
    )
    ax_comp.grid(True, alpha=0.3)
    ax_comp.set_axisbelow(True)
    ax_comp.legend(ncol=2)

    fig_comp.tight_layout()

    plt.show()




if __name__ == "__main__":
    main()