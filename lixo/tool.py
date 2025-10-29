import numpy as np

def generate_mole_fractions(n, num_points=100):
    k = num_points - 1 
    indices = np.array(list(multichoose(n, k))) 
    fractions = indices / k 
    return fractions

def multichoose(n, k):
    if n == 1:
        yield (k,)
    else:
        for i in range(k + 1):
            for tail in multichoose(n - 1, k - i):
                yield (i,) + tail

def gerar_matriz_fracoes_massicas(N, h):
    # Criando discretização de valores entre 0 e 1
    valores = np.linspace(0, 1, h)
    
    # Inicializando lista para armazenar combinações válidas
    matriz = []
    
    # Gerando todas as combinações possíveis para N componentes
    for combinacao in np.ndindex(*([h] * N)):
        fracoes = [valores[i] for i in combinacao]
        if np.isclose(sum(fracoes), 1):  # Verifica se a soma é aproximadamente igual a 1
            matriz.append(fracoes)
    
    return np.array(matriz)


if __name__ == '__main__':
    N = 3
    h = 5
    x = gerar_matriz_fracoes_massicas(N, h)
    print(x)