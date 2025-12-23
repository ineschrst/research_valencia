# -------------------- Standard Libraries -------------------- #
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import time
import io
import random
from collections import defaultdict

# -------------------- Data Handling -------------------- #
import numpy as np
import pandas as pd
from mat4py import loadmat

# -------------------- Plotting -------------------- #
import matplotlib.pyplot as plt
import networkx as nx

# -------------------- ML / SKLearn -------------------- #
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, silhouette_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import NuSVR
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree

# -------------------- Time Series -------------------- #
from statsmodels.tsa.stattools import acf

# -------------------- TensorFlow / Keras -------------------- #
import tensorflow as tf
import keras # This ensures Keras 3 is initialized
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, TimeDistributed, GlobalAveragePooling1D, Reshape
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.losses import Huber
from keras.utils import to_categorical



def to_log(s, fout):
    fout.write(s)
    fout.write('\n')

def show_xy(X, Y, ap, nsamples, nplots=5):
    """
    Plot a few samples of X and Y signals from APSignal data.

    Parameters:
        X : np.array
            Input signals (samples x timesteps)
        Y : np.array
            Output signals (samples x timesteps)
        ap : APSignal
            APSignal object containing timing info
        nsamples : int
            Number of timesteps in X/Y
        nplots : int
            Number of samples to plot (default 5)
    """
    for k in range(nplots):
        plt.figure()
        # Sample indices for plotting
        x_indices = np.linspace(0, nsamples - 1, ap.xtimes[k]).astype(int)
        y_indices = np.linspace(0, nsamples - 1, ap.ytimes[k]).astype(int)

        x_signal = X[k, x_indices]
        y_signal = Y[k, y_indices]

        plt.plot(x_signal, label='X')
        plt.plot(range(len(x_signal) - 1, len(x_signal) + len(y_signal) - 1), y_signal, label='Y')
        plt.legend()
        plt.show()

def show_predictions(testY, testPredict, ap, predTimes, nsamples, split, nplots=5):
    """
    Plot true vs predicted signals for a subset of samples.

    Parameters:
        testY : np.array
            True signals (samples x timesteps)
        testPredict : np.array
            Predicted signals (samples x timesteps)
        ap : APSignal
            APSignal object containing timing info
        predTimes : list or array
            Predicted times for each sample
        nsamples : int
            Number of timesteps to consider
        split : int
            Index offset for sample selection
        nplots : int
            Number of samples to plot (default 5)
    """
    for k in range(nplots):
        y_indices = np.linspace(0, nsamples - 1, ap.ytimes[split + k]).astype(int)
        pred_indices = np.linspace(0, nsamples - 1, int(predTimes[k])).astype(int)

        plt.figure(figsize=(8, 4))
        plt.plot(testY[k, y_indices], c='b', label='Signal')
        plt.plot(testPredict[k, pred_indices], c='g', label='MO-KNNReg Prediction')
        plt.ylim([-85, 40])
        plt.xlabel("msec")
        plt.ylabel("mV")
        plt.legend()
        plt.show()
        plt.close()

def error_plots(testY, testPredict, nsamples, nplots=5):
    """
    Plot error between true and predicted signals.

    Parameters:
        testY : np.array
            True signals (samples x timesteps)
        testPredict : np.array
            Predicted signals (samples x timesteps)
        nsamples : int
            Number of timesteps to consider
        nplots : int
            Number of individual plots to show (default 5)
    """
    # Compute per-timestep error
    err = testY[:, :nsamples] - testPredict[:, :nsamples]
    err = err.T  # shape: (timesteps x samples)
    x = np.arange(nsamples)
    mean_err = np.mean(err, axis=1)
    std_dev = np.std(err, axis=1)

    # Plot error for first nplots samples
    for i in range(nplots):
        signal = testY[i, :nsamples]
        plt.figure(figsize=(8, 4))
        plt.plot(signal, c='b', label='Signal')
        plt.fill_between(x, signal - std_dev, signal + std_dev, color='red', alpha=0.4)
        plt.ylim([-85, 40])
        plt.xlabel("msec")
        plt.ylabel("mV")
        plt.legend()
        plt.show()
        plt.close()

    # Plot mean error across all samples
    plt.figure(figsize=(8, 4))
    plt.plot(mean_err, c='b', label='Mean Error')
    plt.fill_between(x, mean_err - std_dev, mean_err + std_dev, color='red', alpha=0.4)
    plt.xlabel("msec")
    plt.ylabel("mV")
    plt.legend()
    plt.show()
    plt.close()

def create_regression_model():
    """
    Create a multi-output regression model for signal prediction.

    Returns:
        MultiOutputRegressor: A multi-output KNN regressor wrapped for handling multiple outputs.
    """
    # -------------------- Base Regressors -------------------- #
    rf = RandomForestRegressor(n_estimators=100, n_jobs=6)  # Fast, slightly worse than KNN
    knn = KNeighborsRegressor(n_neighbors=1, n_jobs=6)      # Fast and accurate
    svmr = NuSVR(nu=0.2, kernel='rbf')                      # SVM regressor
    lr = LinearRegression(n_jobs=6)                          # Linear regression
    kr = KernelRidge(alpha=0.1, kernel='rbf')               # Very fast, moderate performance
    dt = DecisionTreeRegressor()                             # Smooth output if smode > 0

    # -------------------- Multi-output Wrapper -------------------- #
    mor = MultiOutputRegressor(knn)  # Use KNN as base for multi-output

    return mor

class APSignal:
    """
    A class to process action potential (AP) signal data for analysis,
    including pulse segmentation, APD computation, PCA + clustering to
    create a symbolic alphabet, and sliding-window matrix creation.
    """

    def __init__(self, fname, t_tol, v_tol=-1, threshold=-1, normalize=False, 
                 show=True, mode=0, fout=None):
        """
        Initialize the APSignal instance.
        """
        self.t_tol = t_tol
        self.v_tol = v_tol
        self.threshold = threshold
        self.mode = mode
        self.fout = fout if fout else open("alpha_results.txt", "a")

        # Load the raw signal
        self.raw_data = self._load_signal(fname)
        self.ap = self.raw_data.copy()
        self.size = len(self.ap)

        # Normalize if requested
        if normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.ap = self.scaler.fit_transform(self.ap.reshape(-1,1)).flatten()
            self.threshold = self.scaler.transform([[self.threshold]])[0][0]

        print(f"{fname} loaded. Signal size: {self.ap.shape}")

        # Segment pulses and DI intervals
        self.segment(mode=mode)

        # Compute APDs for pulses
        self.Mapds = np.array([self.get_apds(pulse, show=False)[0] for pulse in self.Mpulses])

        # Build KDTree for fast nearest neighbor search
        self.kdt = KDTree(self.Mapds)

        # Plot raw signal if requested
        if show:
            plt.figure(figsize=(12,4))
            plt.plot(self.ap[:50000])
            plt.title('Raw Pulses')
            plt.show()

        # Initialize placeholders for clustering
        self.alphabet = None
        self.alpha2pulse = None
        self.centroids = None

    # ------------------------ Data Loading ------------------------ #
    def _load_signal(self, fname):
        """Load data from .mat, .npy, or directory of .mat files."""
        if fname.endswith(".mat"):
            data = loadmat(fname)
            return np.array(data['Vm']).flatten()
        elif fname.endswith(".npy"):
            return np.load(fname).flatten()
        elif os.path.isdir(fname):
            return self._load_data_dir_mat(fname)
        else:
            raise ValueError("Unsupported file format. Use .mat, .npy, or directory.")

    def _load_data_dir_mat(self, path):
        raw = []
        for fname in os.listdir(path):
            if fname.endswith(".mat"):
                data = loadmat(os.path.join(path, fname))
                raw.append(np.array(data['Vm']).flatten())
        return np.concatenate(raw)

    # ------------------------ Pulse Segmentation ------------------------ #
    def segment(self, mode=0):
        """
        Segment pulses and DIs from the AP signal.
        """
        Xpulses, Xdis = [], []
        minima = np.unique(np.where(self.ap < self.threshold)[0])
        points = [minima[0]]
        for i in range(1, len(minima)):
            if minima[i] != minima[i-1] + 1:
                points.append(minima[i-1])
                points.append(minima[i])
        max_apd = 0
        for i in range(1, len(points)-2, 2):
            pulso_ini, pulso_fin, pulso_ini2 = points[i], points[i+1], points[i+2]
            if mode == 0:
                x = self.ap[pulso_ini:pulso_fin]
            else:
                x = self.ap[pulso_ini:pulso_ini2]
            di = pulso_ini2 - pulso_fin
            Xpulses.append(x - self.threshold)
            Xdis.append(di)
            max_apd = max(max_apd, len(x))
        self.Mdis = Xdis
        self.rawPulses = Xpulses
        self.max_apd = max_apd
        self.Mpulses = np.zeros([len(Xpulses), self.max_apd + 10])
        for i, pulse in enumerate(Xpulses):
            self.Mpulses[i,:len(pulse)] = pulse - pulse[0]

    # ------------------------ APD Computation ------------------------ #
    def get_apds(self, voltage, show=False):
        """
        Compute APD values for a pulse.
        """
        deriv = np.diff(voltage)
        t_start = np.argmax(deriv)
        percentages = [10,20,30,50,70,80,90,100]
        apd_results = [t_start, np.argmax(voltage)]
        for pct in percentages:
            vmax, vmin = np.max(voltage), np.min(voltage)
            target = vmin + (vmax - vmin) * (1 - pct/100)
            crossings = np.where(np.diff(np.sign(voltage - target)))[0]
            if len(crossings) >= 1:
                apd_results.append(crossings[-1])
            else:
                apd_results.append(None)
        return apd_results, voltage[apd_results]

    # ------------------------ Pulse Clustering ------------------------ #
    def pulses2alphabet(self, ncomponents=12, nclusters=10, show=False):
        """
        Apply PCA and KMeans clustering to generate symbolic alphabet.
        """
        X_reduced = PCA(n_components=ncomponents).fit_transform(self.Mpulses) if ncomponents>1 else self.Mapds
        kmeans = KMeans(n_clusters=nclusters, random_state=42)
        labels = kmeans.fit_predict(X_reduced)
        centroids_reduced = kmeans.cluster_centers_
        centroids = PCA(n_components=ncomponents).fit(self.Mpulses).inverse_transform(centroids_reduced)
        self.alphabet = labels
        self.centroids = centroids
        self.alpha2pulse = {i: centroids[i] for i in range(nclusters)}

    # ------------------------ Sliding Window Matrices ------------------------ #
    def crear_matrices(self, look_back):
        """
        Create X, Y matrices using a sliding window from alphabet.
        """
        X, Y = [], []
        for i in range(0, len(self.alphabet)-look_back, 2):
            X.append(self.alphabet[i:i+look_back])
            Y.append(self.alphabet[i+look_back])
        return np.array(X), np.array(Y)
    
def plot_pca_variance(M):
    """
    Plot explained variance and cumulative variance for PCA components.

    Args:
        M (np.ndarray): Input data matrix (samples x features).
    """
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(M)

    # Fit PCA
    pca = PCA()
    pca.fit(X_scaled)

    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    # Plot cumulative variance and individual variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance)+1), cumulative_variance,
             marker='o', linestyle='--', color='blue', label='Cumulative variance')
    plt.bar(range(1, len(explained_variance)+1), explained_variance,
            alpha=0.5, color='orange', label='Individual explained variance')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.xticks(range(1, len(explained_variance)+1))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def optim_k(X, mode=0, krange=range(2, 20)):
    """
    Determine the optimal number of clusters using either the elbow or silhouette method.

    Args:
        X (np.ndarray): Data to cluster (samples x features)
        mode (int): 0 for elbow method (inertia), 1 for silhouette method
        krange (iterable): Range of k (number of clusters) to test
    """
    if mode == 0:
        # Elbow method: inertia
        inertias = []
        for k in krange:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

        # Relative change in inertia
        rel_changes = np.abs(np.diff(inertias) / inertias[:-1])

        # Plot relative change
        plt.figure(figsize=(8, 5))
        plt.plot(krange[1:], rel_changes, marker='o', linestyle='--', color='orange')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Relative change in inertia")
        plt.title("Elbow Method - Relative Change in Inertia")
        plt.grid(alpha=0.3)
        plt.show()

        # Plot inertia vs k
        plt.figure(figsize=(8, 5))
        plt.plot(krange, inertias, marker='o', linestyle='--', color='blue')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Inertia (WCSS)")
        plt.title("Elbow Method for Determining k")
        plt.grid(alpha=0.3)
        plt.show()

    else:
        # Silhouette method
        silhouette_scores = []
        for k in krange:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X, labels))

        plt.figure(figsize=(8, 5))
        plt.plot(krange, silhouette_scores, marker='o', linestyle='--', color='green')
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Method for Determining k")
        plt.grid(alpha=0.3)
        plt.show()


# Valores únicos y sus frecuencias
def show_labels_hist(labels):
    values, counts = np.unique(labels, return_counts=True)

    plt.bar(values, counts, color='skyblue', edgecolor='black')

    # Configuración del gráfico
    plt.title("Distribución de Frecuencias", fontsize=16)
    plt.xlabel("Valores Únicos", fontsize=14)
    plt.ylabel("Frecuencia", fontsize=14)
    plt.xticks(values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Mostrar los valores encima de las barras
    for i in range(len(values)):
        plt.text(values[i], counts[i] + 0.1, str(counts[i]), ha='center', fontsize=12)

    # Mostrar la gráfica
    plt.tight_layout()
    plt.show()


def convertir_lista_a_grafo(lista):
    # Contar transiciones
    transiciones = defaultdict(lambda: defaultdict(int))
    for i in range(len(lista) - 1):
        origen = lista[i]
        destino = lista[i + 1]
        transiciones[origen][destino] += 1

    # Calcular probabilidades
    grafo = {}
    for origen, destinos in transiciones.items():
        total_transiciones = sum(destinos.values())
        grafo[origen] = {destino: count / total_transiciones for destino, count in destinos.items()}

    return grafo

def graficar_grafo(grafo):
    G = nx.DiGraph()
    
    # Añadir nodos y aristas con probabilidades como peso
    for origen, destinos in grafo.items():
        for destino, probabilidad in destinos.items():
            G.add_edge(origen, destino, weight=probabilidad)
    
    # Dibujar el grafo
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold', arrowsize=20)
    
    # Añadir etiquetas de peso
    etiquetas = nx.get_edge_attributes(G, 'weight')
    etiquetas = {k: f"{v:.2f}" for k, v in etiquetas.items()}  # Formato de dos decimales
    nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquetas, font_size=8)
    
    plt.show()


def generar_cadena(grafo, estado_anterior, estado_actual, k):
    """
    Genera una cadena de longitud k a partir de un estado inicial y un estado anterior basado en las probabilidades del grafo.

    :param grafo: Diccionario de probabilidades condicionadas, e.g., {(estado_previo, estado_actual): {estado_siguiente: probabilidad}}.
    :param estado_anterior: Estado anterior inicial del recorrido.
    :param estado_actual: Estado actual inicial del recorrido.
    :param k: Longitud de la cadena a generar.
    :return: Lista de estados generados.
    """
    cadena = [estado_anterior, estado_actual]  # Empezamos con los dos estados iniciales

    for _ in range(k - 2):  # Ya tenemos dos elementos, generamos los restantes
        # Buscar transiciones basadas en el par (estado_anterior, estado_actual)
        posibles_transiciones = grafo.get((estado_anterior, estado_actual), None)

        if not posibles_transiciones:
            # Si no hay transiciones posibles, terminamos el recorrido
            break

        # Elegir el siguiente estado basado en las probabilidades
        estados = list(posibles_transiciones.keys())
        probabilidades = list(posibles_transiciones.values())
        estado_siguiente = random.choices(estados, weights=probabilidades, k=1)[0]

        # Actualizar la cadena y los estados
        cadena.append(estado_siguiente)
        estado_anterior = estado_actual
        estado_actual = estado_siguiente

    return cadena

# Función para generar el grafo (para referencia)
def convertir_lista_a_grafo_condicionado(lista):
    from collections import defaultdict
    transiciones = defaultdict(lambda: defaultdict(int))
    for i in range(len(lista) - 2):
        estado_previo = lista[i]
        estado_actual = lista[i + 1]
        estado_siguiente = lista[i + 2]
        transiciones[(estado_previo, estado_actual)][estado_siguiente] += 1

    grafo = {}
    for (estado_previo, estado_actual), destinos in transiciones.items():
        total_transiciones = sum(destinos.values())
        grafo[(estado_previo, estado_actual)] = {
            destino: count / total_transiciones for destino, count in destinos.items()
        }

    return grafo

def graficar_grafo_condicionado(grafo):
    G = nx.DiGraph()

    # Añadir nodos y aristas con probabilidades como peso
    for (estado_previo, estado_actual), destinos in grafo.items():
        for estado_siguiente, probabilidad in destinos.items():
            G.add_edge(estado_actual, estado_siguiente, weight=probabilidad, label=f"{estado_previo}->{estado_actual}->{estado_siguiente}")

    # Dibujar el grafo
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=200, font_size=10, font_weight='bold', arrowsize=10)

    # Añadir etiquetas de peso
    etiquetas = nx.get_edge_attributes(G, 'weight')
    #etiquetas = {k: f"{v:.2f}" for k, v in etiquetas.items()}  # Formato de dos decimales
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=etiquetas, font_size=8)

    plt.show()

def construir_grafo_desde_aristas(estados, aristas):
    """
    Construye un grafo dirigido a partir de una lista de estados y una lista de aristas.

    :param estados: Lista de enteros que representan los nodos del grafo.
    :param aristas: Lista de enteros que representan los valores (pesos) de las aristas entre estados consecutivos.
    :return: Diccionario que representa el grafo, con pesos asociados a las transiciones.
    """
    if len(aristas) != len(estados) - 1:
        raise ValueError("El número de aristas debe ser igual al número de transiciones posibles (len(estados) - 1).")
    
    grafo = {}
    for i in range(len(aristas)):
        origen = estados[i]
        destino = estados[i + 1]
        peso = aristas[i]
        
        if origen not in grafo:
            grafo[origen] = {}
        grafo[origen][destino] = peso

    return grafo

def graficar_grafo_di(grafo):
    """
    Grafica un grafo dirigido utilizando NetworkX y Matplotlib.
    
    :param grafo: Diccionario que representa el grafo, con pesos asociados a las transiciones.
    """
    G = nx.DiGraph()
    
    # Agregar nodos y aristas al grafo
    for origen, destinos in grafo.items():
        for destino, peso in destinos.items():
            G.add_edge(origen, destino, weight=peso)
    
    # Obtener pesos para etiquetar las aristas
    edge_labels = nx.get_edge_attributes(G, 'weight')
    
    # Dibujar el grafo
    pos = nx.spring_layout(G)  # Layout para los nodos
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=200, font_size=10, font_weight='bold', arrowsize=15)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    
    plt.title("Grafo Dirigido")
    plt.show()

def recorrer_grafo(grafo, estado_inicial, pasos):
    """
    Realiza un recorrido por el grafo dirigido y devuelve los estados visitados y los pesos de las aristas recorridas.

    :param grafo: Diccionario que representa el grafo, con pesos asociados a las transiciones.
    :param estado_inicial: Nodo de inicio del recorrido.
    :param pasos: Número de pasos a realizar en el recorrido.
    :return: Tuple (lista_estados, lista_pesos) donde:
             - lista_estados: Lista de estados visitados (incluyendo el inicial).
             - lista_pesos: Lista de pesos de las aristas recorridas.
    """
    estados_visitados = [estado_inicial]  # Inicializar con el estado inicial
    pesos_aristas = []

    estado_actual = estado_inicial

    for _ in range(pasos):
        if estado_actual not in grafo or not grafo[estado_actual]:
            # Si no hay transiciones desde el estado actual, termina el recorrido
            break
        
        # Elegir el siguiente estado basado en las aristas disponibles
        destinos = list(grafo[estado_actual].keys())
        pesos = list(grafo[estado_actual].values())
        
        # Seleccionar un destino aleatorio ponderado por los pesos
        estado_siguiente = random.choices(destinos, weights=pesos, k=1)[0]
        
        # Obtener el peso de la arista seleccionada
        peso_arista = grafo[estado_actual][estado_siguiente]
        
        # Actualizar las listas
        estados_visitados.append(estado_siguiente)
        pesos_aristas.append(peso_arista)
        
        # Moverse al siguiente estado
        estado_actual = estado_siguiente

    return estados_visitados, pesos_aristas

# 2. Entrenar un modelo de clasificación multiclase
def entrenar_modelo(X, Y):
    """
    Entrena un modelo de clasificación con los datos dados.
    
    Args:
        X (np.array): Matriz de características.
        Y (np.array): Vector de etiquetas objetivo.
    
    Returns:
        modelo: Modelo entrenado.
    """
    from sklearn.neighbors import KNeighborsClassifier

#    modelo = KNeighborsClassifier(n_neighbors=1)
    modelo = RandomForestClassifier(random_state=42)
    #print("Yeee ", X.shape, Y.shape)
    modelo.fit(X, Y)
    return modelo

# 3. Evaluar el modelo entrenado
def evaluar_modelo(modelo, X_test, Y_test):
    """
    Evalúa el modelo con un conjunto de test y muestra los resultados.
    
    Args:
        modelo: Modelo entrenado.
        X_test (np.array): Matriz de características de test.
        Y_test (np.array): Vector de etiquetas objetivo de test.
    """
    
    Y_pred = modelo.predict(X_test)
    print("Accuracy:", accuracy_score(Y_test, Y_pred))
    #print("Reporte de Clasificación:\n", classification_report(Y_test, Y_pred))

class HP_model:
    """
    Class to define, train, and evaluate pulse signal models.
    Supports multiple model types: Random Forest (classification), LSTM, GRU, CNN.
    """

    def __init__(self, ap_sig, mode='reg', look_back=128, hp=13, nclusters=14, ncomponents=12, alpha=0.5, fout=open("alpha_results.txt","a")):
        self.ap = ap_sig
        self.look_back = look_back
        self.hp = hp
        self.model_type = mode
        self.nclusters = nclusters
        self.ncomponents = ncomponents
        self.alpha = alpha
        self.fout = fout

        # Generate alphabet and pulse tensor
        self.ap.pulses2alphabet(nclusters=nclusters, ncomponents=ncomponents, show=False)
        self.alpha2pulse_tensor = tf.convert_to_tensor(list(self.ap.alpha2pulse.values()), dtype=tf.float32)
        self.alphabet = self.ap.alphabet.reshape(-1, 1)

        # Initialize model
        self.model, self.callback = self._initialize_model(mode)

    def _initialize_model(self, mode):
        """Initialize the model based on the specified type."""
        if mode == 'reg':
            return self.create_classification_model(), None
        elif mode == 'lstm':
            return self.create_lstm_model(self.alpha)
        elif mode == 'gru':
            return self.create_gru_model(self.alpha)
        elif mode == 'cnn':
            return self.create_cnn_model(self.alpha)
        else:
            raise ValueError(f"Unknown model type: {mode}")

    # ------------------------
    # Dataset building methods
    # ------------------------
    def build_data_set(self):
        """
        Build training and testing datasets from the alphabet sequence.
        Supports LSTM/GRU/CNN reshaping automatically.
        """
        split_idx = int(0.9 * len(self.alphabet))
        alphabet_train, alphabet_test = self.alphabet[:split_idx], self.alphabet[split_idx:]
        
        X_train, Y_train = [], []
        for i in range(len(alphabet_train) - self.look_back - self.hp):
            X_train.append(alphabet_train[i:i+self.look_back])
            Y_train.append(alphabet_train[i+self.look_back:i+self.look_back+self.hp].flatten())

        X_test, Y_test, test_indices = [], [], []
        for i in range(len(alphabet_test) - self.look_back - self.hp):
            X_test.append(alphabet_test[i:i+self.look_back])
            Y_test.append(alphabet_test[i+self.look_back:i+self.look_back+self.hp].flatten())
            test_indices.append(range(split_idx + i + self.look_back, split_idx + i + self.look_back + self.hp))

        # Convert to arrays
        X_train, Y_train = np.array(X_train), np.array(Y_train)
        X_test, Y_test = np.array(X_test), np.array(Y_test)
        
        # Reshape for RNN/CNN
        if self.model_type in ['lstm', 'gru', 'cnn']:
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], self.nclusters)
            Y_train = Y_train.reshape(Y_train.shape[0], self.hp, self.nclusters)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], self.nclusters)
            Y_test = Y_test.reshape(Y_test.shape[0], self.hp, self.nclusters)

        self.test_indices = test_indices
        return X_train, Y_train, X_test, Y_test

    # ------------------------
    # Model creation methods
    # ------------------------
    def create_classification_model(self):
        """Random Forest classification model."""
        return RandomForestClassifier(n_estimators=200, n_jobs=-1)

    def get_custom_loss(self, alpha2pulse_tensor, alpha):
        """Return a combined loss function (categorical + custom pulse loss)."""
        def loss_fn(y_true, y_pred):
            # Categorical crossentropy
            cce_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)
            # Custom loss: weighted pulse reconstruction
            Y_pred_probs = tf.nn.softmax(y_pred, axis=-1)
            Y_true_probs = tf.nn.softmax(y_true, axis=-1)
            decoded_pred = tf.tensordot(Y_pred_probs, alpha2pulse_tensor, axes=[[2], [0]])
            decoded_true = tf.tensordot(Y_true_probs, alpha2pulse_tensor, axes=[[2], [0]])
            custom_loss = tf.reduce_mean(tf.reduce_mean(tf.abs(decoded_pred - decoded_true), axis=-1), axis=-1)
            return alpha * cce_loss + (1 - alpha) * custom_loss
        return loss_fn

    def create_lstm_model(self, alpha):
        """LSTM model for multi-class pulse prediction."""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.look_back, self.nclusters)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.nclusters*self.hp, activation='softmax'),
            Reshape((self.hp, self.nclusters))
        ])
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
        return model, callback

    def create_gru_model(self, alpha):
        """GRU model for multi-class pulse prediction."""
        model = Sequential([
            GRU(64, return_sequences=True, input_shape=(self.look_back, self.nclusters)),
            Dropout(0.2),
            GRU(32, return_sequences=False),
            Dropout(0.2),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.nclusters*self.hp, activation='softmax'),
            Reshape((self.hp, self.nclusters))
        ])
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        return model, callback

    def create_cnn_model(self, alpha):
        """1D CNN model for multi-class pulse prediction."""
        model = Sequential([
            Conv1D(32, kernel_size=3, activation='relu', input_shape=(self.look_back, self.nclusters)),
            MaxPooling1D(2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(2),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(2),
            GlobalAveragePooling1D(),
            Dense(128, activation='relu'),
            Dropout(0.2),
            Dense(self.nclusters*self.hp, activation='softmax'),
            Reshape((self.hp, self.nclusters))
        ])
        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        return model, callback

    # ------------------------
    # Testing and evaluation
    # ------------------------
    def test_reg(self, fout, X_train, Y_train, X_test, Y_test):
        """Evaluate Random Forest classification model."""
        X_train, X_test = X_train.reshape(X_train.shape[0], -1), X_test.reshape(X_test.shape[0], -1)
        start_time = time.time()
        self.model.fit(X_train, Y_train)
        elapsed = (time.time() - start_time) / 60
        to_log(f"Fitting time: {elapsed:.4f} minutes", fout)
        Y_pred = self.model.predict(X_test)
        accuracy = np.mean(Y_pred == Y_test)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        to_log(f"Test Accuracy: {accuracy*100:.2f}%", fout)
        return accuracy

    def test_lstm_gru(self, fout, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=32):
        """Evaluate RNN/CNN model with categorical accuracy."""
        X_train_cat = to_categorical(X_train, num_classes=self.nclusters)
        X_test_cat = to_categorical(X_test, num_classes=self.nclusters)
        Y_train_cat = to_categorical(Y_train, num_classes=self.nclusters)
        Y_test_cat = to_categorical(Y_test, num_classes=self.nclusters)

        start_time = time.time()
        history = self.model.fit(X_train_cat, Y_train_cat, validation_split=0.05,
                                 epochs=epochs, batch_size=batch_size, verbose=1,
                                 callbacks=[self.callback])
        elapsed = (time.time() - start_time)/60
        to_log(f"Fitting time: {elapsed:.4f} minutes", fout)

        Y_pred = np.argmax(self.model.predict(X_test_cat), axis=-1)
        Y_true = np.argmax(Y_test_cat, axis=-1)
        accuracy = np.mean(Y_pred == Y_true)
        print(f"Test Accuracy: {accuracy*100:.2f}%")
        to_log(f"Test Accuracy: {accuracy*100:.2f}%", fout)
        return accuracy

    # ------------------------
    # Utility
    # ------------------------
    def plot_loss(self, history):
        """Plot training and validation loss."""
        plt.figure(figsize=(8,5))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()  

##########################################################
# def compare_hp_old(ap, fout, lback = 128, hp=30, plot_loss=False, epochs=100, batch_size=32, plot_accuracy=False, nclusters=10):
   
#     model_reg = HP_model(ap, mode='reg', lback = lback, hp=hp, nclusters=nclusters)
#     al2p_reg =  model_reg.alpha2pulse_tensor
#     X_reg_train, Y_reg_train, X_reg_test, Y_reg_test = model_reg.build_data_set()
#     model_lstm = HP_model(ap, mode='lstm', lback = lback, hp=hp, nclusters=nclusters)
#     al2p_lstm =  model_lstm.alpha2pulse_tensor
#     X_lstm_train, Y_lstm_train, X_lstm_test, Y_lstm_test = model_lstm.build_data_set()
#     model_gru = HP_model(ap, mode='gru', lback = lback, hp=hp, nclusters=nclusters)
#     al2p_gru =  model_gru.alpha2pulse_tensor
#     X_gru_train, Y_gru_train, X_gru_test, Y_gru_test = model_gru.build_data_set()
#     model_cnn = HP_model(ap, mode='cnn', lback = lback, hp=hp, nclusters=nclusters)
#     al2p_cnn =  model_cnn.alpha2pulse_tensor
#     X_cnn_train, Y_cnn_train, X_cnn_test, Y_cnn_test = model_cnn.build_data_set()

#     to_log("Random Forest", fout)
#     err_reg = model_reg.test_reg(fout, X_reg_train, Y_reg_train, X_reg_test, Y_reg_test, plot_loss=plot_loss, plot_accuracy=plot_accuracy)
#     to_log("LSTM", fout)
#     err_lstm = model_lstm.test_lstm_gru(fout, X_lstm_train, Y_lstm_train, X_lstm_test, Y_lstm_test, plot_loss=plot_loss, epochs=epochs, batch_size=batch_size, plot_accuracy=plot_accuracy)
#     to_log("GRU", fout)
#     err_gru = model_gru.test_lstm_gru(fout, X_gru_train, Y_gru_train, X_gru_test, Y_gru_test, plot_loss=plot_loss, epochs=epochs, batch_size=batch_size, plot_accuracy=plot_accuracy)
#     to_log("CNN", fout)
#     err_cnn = model_cnn.test_lstm_gru(fout, X_cnn_train, Y_cnn_train, X_cnn_test, Y_cnn_test, plot_loss=plot_loss, epochs=epochs, batch_size=batch_size, plot_accuracy=plot_accuracy)
    
#     fig = plt.figure(figsize=(18,2))
#     for i in range(len(err_reg)):
#         plt.scatter(i, err_reg[i], color='xkcd:sky blue') #, c='b')
#         if i == 0:
#             plt.plot(err_reg[i], label='Random Forest', color='xkcd:sky blue')
#         plt.plot(err_reg[i], color='xkcd:sky blue')
#         print(f"Accuracy Horizon Prediction:  {err_reg[i]:.5f} Random Forest")
#     for i in range(len(err_lstm)):
#         plt.scatter(i, err_lstm[i], color='xkcd:pink') #, c='b')
#         if i == 0:
#             plt.plot(err_lstm[i], label='LSTM', color='xkcd:pink')
#         plt.plot(err_lstm[i], color='xkcd:pink')
#         print(f"Accuracy Horizon Prediction: {err_lstm[i]:.5f} LSTM")
#     for i in range(len(err_gru)):
#         plt.scatter(i, err_gru[i], color='xkcd:light green') #, c='b')
#         if i == 0:
#             plt.plot(err_gru[i], label='GRU', color='xkcd:light green')
#         plt.plot(err_gru[i], color='xkcd:light green')
#         print(f"Accuracy Horizon Prediction:  {err_gru[i]:.5f} GRU")
#     for i in range(len(err_cnn)):
#         plt.scatter(i, err_cnn[i], color='xkcd:lavender')
#         if i == 0:
#             plt.plot(err_cnn[i], label='CNN', color='xkcd:lavender')
#         plt.plot(err_cnn[i], color='xkcd:lavender')
#         print(f"Accuracy Horizon Prediction:  {err_cnn[i]:.5f} CNN")
#     plt.title('Accuracies of different tries for different models')
#     plt.ylabel('Accuracy')
#     plt.xlabel('tries')
    
#     combined_list = [(arr, "Random Forest") for arr in err_reg] + [(arr, "LSTM") for arr in err_lstm] + [(arr, "GRU") for arr in err_gru] + [(arr, "CNN") for arr in err_cnn]
#     lowest_mean_entry = max(combined_list, key=lambda entry: np.mean(entry[0]))
#     lowest_mean_array, list_name = lowest_mean_entry
#     print("Highest Accuracy:", np.mean(lowest_mean_array))
#     print("Network:", list_name)
 
#     X_test_reg = X_reg_test.reshape(X_reg_test.shape[0], -1) 
#     Y_pred_reg = model_reg.model.predict(X_test_reg)
    
#     X_test_one_hot_lstm = to_categorical(X_lstm_test, num_classes=model_lstm.nclusters)
#     Y_pred_lstm = model_lstm.model.predict(X_test_one_hot_lstm)
    
#     X_test_one_hot_gru = to_categorical(X_gru_test, num_classes=model_gru.nclusters)
#     Y_pred_gru = model_gru.model.predict(X_test_one_hot_gru)

#     X_test_one_hot_cnn = to_categorical(X_cnn_test, num_classes=model_cnn.nclusters)
#     Y_pred_cnn = model_cnn.model.predict(X_test_one_hot_cnn)

#     print(np.array(Y_reg_test).shape, np.array(Y_pred_reg).shape)
#     print(np.array(Y_lstm_test).shape, np.array(Y_pred_lstm).shape)
#     print(np.array(Y_gru_test).shape, np.array(Y_pred_gru).shape)
#     print(np.array(Y_cnn_test).shape, np.array(Y_pred_cnn).shape)
    
#     overall_error_reg = overall_loss(Y_reg_test, Y_pred_reg, fout, al2p_reg, model_type='reg')
#     overall_error_lstm = overall_loss(Y_lstm_test, Y_pred_lstm, fout, al2p_lstm, model_type='lstm')
#     overall_error_gru = overall_loss(Y_gru_test, Y_pred_gru, fout, al2p_gru, model_type='gru')
#     overall_error_cnn = overall_loss(Y_cnn_test, Y_pred_cnn, fout,  al2p_cnn, model_type='cnn')

#     print('X test shape', X_test_reg.shape)
#     print('X test shape', X_test_one_hot_lstm.shape)
#     apds_pred(Y_reg_test, Y_pred_reg, al2p_reg, model_type='reg')
#     apds_pred(Y_lstm_test, Y_pred_lstm, al2p_lstm, model_type='lstm')
    
#     print("The error between predicted and true pulses, RandomForest: ", overall_error_reg.numpy())
#     print("The error between predicted and true pulses, LSTM: ", overall_error_lstm.numpy())
#     print("The error between predicted and true pulses, GRU: ", overall_error_gru.numpy())
#     print("The error between predicted and true pulses, CNN: ", overall_error_cnn.numpy())

#     error_values = [(overall_error_reg.numpy(), "Random Forest"), (overall_error_lstm.numpy(), "LSTM"), (overall_error_gru.numpy(), "GRU"),  (overall_error_cnn.numpy(), "CNN")]
#     lowest_error_value, lowest_error_model = min(error_values, key=lambda x: x[0])
#     print(f"The lowest error: {lowest_error_value:.3f} was achieved by the network {lowest_error_model}")
    
#     plt.legend()
#     plt.show()
#     return err_reg, err_lstm, err_gru, np.mean(lowest_mean_array), list_name

def create_and_train_model(ap, fout, mode, look_back, hp, plot_loss, epochs, batch_size, plot_accuracy, nclusters):
    """
    Create, train, and evaluate a model for a given mode.

    Parameters
    ----------
    ap : object
        Input dataset object.
    fout : file object
        File to log output.
    mode : str
        Model type ('reg', 'lstm', 'gru', 'cnn').
    look_back : int
        Lookback window size.
    hp : int
        Prediction horizon.
    plot_loss : bool
        Whether to plot training loss.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    plot_accuracy : bool
        Whether to plot accuracy for individual samples.
    nclusters : int
        Number of clusters.

    Returns
    -------
    model : HP_model
        Trained model object.
    X_test : np.ndarray
        Test features.
    Y_test : np.ndarray
        Test labels.
    error : float or list
        Model error or accuracy.
    """
    to_log(f"{mode.upper()}", fout)
    
    model = HP_model(ap, mode=mode, look_back=look_back, hp=hp, nclusters=nclusters, fout=fout)
    X_train, Y_train, X_test, Y_test = model.build_data_set()
    
    if mode == 'reg':
        error = model.test_reg(fout, X_train, Y_train, X_test, Y_test)
    else:
        error = model.test_lstm_gru(fout, X_train, Y_train, X_test, Y_test,
                                    plot_loss=plot_loss, epochs=epochs, batch_size=batch_size)
    
    return model, X_test, Y_test, error


def calculate_error(model, fout, X_test, Y_test, alpha2pulse_tensor, mode, plot_pulses, Mpulses):
    """
    Calculate errors between true and predicted values.

    Parameters
    ----------
    model : HP_model
        Trained model object.
    X_test : np.ndarray
        Test input features.
    Y_test : np.ndarray
        True labels.
    alpha2pulse_tensor : tf.Tensor
        Mapping from alphabet indices to pulses.
    mode : str
        Model type ('reg', 'lstm', 'gru', 'cnn').
    plot_pulses : bool
        Whether to plot pulse comparisons.
    Mpulses : np.ndarray
        True pulses for comparison.

    Returns
    -------
    overall_error : float
        Loss/error between true and predicted centroids.
    overall_error_2 : float
        Loss/error between true pulses and predicted centroids.
    """
    if mode == 'reg':
        X_test_input = X_test.reshape(X_test.shape[0], -1)
        predictions = model.model.predict(X_test_input)
    else:
        X_test_input = to_categorical(X_test, num_classes=model.nclusters)
        predictions = model.model.predict(X_test_input)
    
    overall_error = overall_loss(Y_test, predictions, fout, alpha2pulse_tensor, model_type=mode, plot_pulses=plot_pulses)
    overall_error_2 = overall_loss_2(Y_test, Mpulses, model.test_indices, predictions, fout, alpha2pulse_tensor, model_type=mode, plot_pulses=plot_pulses)
    
    return overall_error, overall_error_2


def plot_accuracies(errors):
    """
    Plot accuracies for multiple models.

    Parameters
    ----------
    errors : dict
        Dictionary of model errors/accuracies.
    """
    colors = {'reg': 'xkcd:sky blue', 'lstm': 'xkcd:pink', 'gru': 'xkcd:light green', 'cnn': 'xkcd:lavender'}
    labels = {'reg': 'Random Forest', 'lstm': 'LSTM', 'gru': 'GRU', 'cnn': 'CNN'}
    
    plt.figure(figsize=(18, 2))
    for mode, error_list in errors.items():
        plt.plot(range(len(error_list)), error_list, marker='o', color=colors[mode], label=labels[mode])
    
    plt.title('Accuracies of Different Models')
    plt.ylabel('Accuracy')
    plt.xlabel('Tries')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def find_best_model(errors):
    """
    Identify the best-performing model based on mean accuracy.

    Parameters
    ----------
    errors : dict
        Dictionary of model errors/accuracies.

    Returns
    -------
    best_model_name : str
        Name of the best-performing model.
    best_accuracy : float
        Mean accuracy of the best-performing model.
    """
    mean_accuracies = {mode: np.mean(err) for mode, err in errors.items()}
    best_model_name = max(mean_accuracies, key=mean_accuracies.get)
    best_accuracy = mean_accuracies[best_model_name]
    return best_model_name, best_accuracy


def compare_hp(ap, fout, look_back=128, hp=30, plot_loss=False, epochs=100, batch_size=32,
               plot_accuracy=False, plot_pulses=False, nclusters=10):
    """
    Compare performance of multiple models on horizon prediction tasks.

    Parameters
    ----------
    ap : object
        Input dataset object (e.g., APSignal instance).
    look_back : int
        Lookback window size.
    hp : int
        Prediction horizon.
    plot_loss : bool
        Whether to plot training loss curves.
    epochs : int
        Training epochs.
    batch_size : int
        Training batch size.
    plot_accuracy : bool
        Whether to plot individual accuracies.
    plot_pulses : bool
        Whether to plot pulse comparisons.
    nclusters : int
        Number of clusters.

    Returns
    -------
    errors : dict
        Dictionary of errors/accuracies for each model.
    best_accuracy : float
        Mean accuracy of the best-performing model.
    best_model_name : str
        Name of the best-performing model.
    """
    to_log(f"lback={look_back}, hp={hp}, epochs={epochs}, batch_size={batch_size}", fout)
    
    models, X_tests, Y_tests, errors = {}, {}, {}, {}
    for mode in ['reg', 'lstm', 'gru', 'cnn']:
        model, X_test, Y_test, error = create_and_train_model(
            ap, fout, mode, look_back, hp, plot_loss, epochs, batch_size, plot_accuracy, nclusters
        )
        models[mode], X_tests[mode], Y_tests[mode], errors[mode] = model, X_test, Y_test, error
    
    # Plot accuracies
    plot_accuracies(errors)
    
    # Identify best model
    best_model_name, best_accuracy = find_best_model(errors)
    
    # Calculate detailed errors
    Mpulses = ap.Mpulses
    for mode in models:
        overall_error, overall_error_2 = calculate_error(models[mode], fout, X_tests[mode], Y_tests[mode],
                                                          models[mode].alpha2pulse_tensor, mode,
                                                          plot_pulses, Mpulses)
        print(f"Centroid error for {mode.upper()}: {overall_error:.3f}")
        print(f"Pulse error for {mode.upper()}: {overall_error_2:.3f}")
    
    print(f"Best-performing model: {best_model_name.upper()} with mean accuracy {best_accuracy:.3f}")
    
    return errors, best_accuracy, best_model_name

# -----------------------------
# Model Evaluation
# -----------------------------
def evaluate_model(ap, fout, mode='lstm', lback=128, hp=30, steps=1,
                   plot_loss=False, epochs=100, batch_size=32,
                   plot_accuracy=False, plot_pulses=False, apds_pred_plot=False,
                   nclusters=10, sequential=False):
    """
    Generalized model evaluation using HP_model class.
    
    Args:
        ap (APSignal): The APSignal object with pulses and alphabet.
        fout (file-like): File for logging.
        mode (str): Model type ('reg', 'lstm', 'gru', 'cnn').
        lback (int): Look-back window size.
        hp (int): Prediction horizon.
        steps (int): Number of sequential steps (if sequential=True).
        plot_loss (bool): Plot training loss.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        plot_accuracy (bool): Plot accuracy.
        plot_pulses (bool): Plot reconstructed pulses.
        apds_pred_plot (bool): Plot predicted APDs.
        nclusters (int): Number of clusters for alphabet.
        sequential (bool): Whether to use stepwise/sequential prediction.
        
    Returns:
        dict: Evaluation results including error metrics and predictions.
    """
    to_log(f"Evaluating {mode.upper()} | lback={lback}, hp={hp}, steps={steps}, epochs={epochs}", fout)

    # Initialize HP_model
    model = HP_model(ap, mode=mode, look_back=lback, hp=hp, nclusters=nclusters, fout=fout)

    # Build datasets
    X_train, Y_train, X_test, Y_test = model.build_data_set()

    # Train / evaluate model
    if mode == 'reg':
        error = model.test_reg(fout, X_train, Y_train, X_test, Y_test)
    else:
        error = model.test_lstm_gru(fout, X_train, Y_train, X_test, Y_test,
                                    epochs=epochs, batch_size=batch_size)

    # Predictions
    if sequential:
        # Stepwise prediction: call stepwise_prediction function
        Y_pred = stepwise_prediction(model, X_test, steps=steps, look_back=lback,
                                     hp=hp, alpha2pulse_tensor=model.alpha2pulse_tensor,
                                     nclusters=nclusters, mode=mode)
    else:
        # Single-step prediction
        if mode == 'reg':
            Y_pred = model.model.predict(X_test.reshape(X_test.shape[0], -1))
        else:
            X_test_cat = to_categorical(X_test, num_classes=nclusters)
            Y_pred = np.argmax(model.model.predict(X_test_cat), axis=-1)

    # Calculate overall errors
    overall_error = overall_loss(Y_test, Y_pred, fout, model.alpha2pulse_tensor, mode, plot_pulses)
    overall_error_2 = overall_loss_2(Y_test, ap.Mpulses, model.test_indices, Y_pred, fout,
                                    model.alpha2pulse_tensor, mode, plot_pulses, mode=ap.mode)

    # Optional APD visualization
    if apds_pred_plot:
        plot_apds(Y_test, Y_pred, model.alpha2pulse_tensor, mode)

    return {
        'model_type': mode,
        'error': error,
        'overall_error': overall_error.numpy() if hasattr(overall_error, 'numpy') else overall_error,
        'overall_error_2': overall_error_2.numpy() if hasattr(overall_error_2, 'numpy') else overall_error_2,
        'predictions': Y_pred,
        'true_values': Y_test
    }


# -----------------------------
# Prediction Helpers
# -----------------------------
def predict(model, X_test, mode):
    """Predict using trained model."""
    if mode == 'reg':
        X_input = X_test.reshape(X_test.shape[0], -1)
        return model.model.predict(X_input)
    else:
        X_input = to_categorical(X_test, num_classes=model.nclusters)
        return model.model.predict(X_input)


def stepwise_prediction(model, X_initial, steps, look_back, hp, alpha2pulse_tensor, nclusters, mode):
    """Perform stepwise prediction over the horizon."""
    predictions = []
    current_input = X_initial.copy()

    for _ in range(int(hp / steps)):
        if mode == 'reg':
            X_input = current_input.reshape(current_input.shape[0], -1)
            y_pred = model.model.predict(X_input)
            y_pred = np.expand_dims(y_pred, axis=-1)
        else:
            X_input = to_categorical(current_input, num_classes=nclusters)
            y_pred = model.model.predict(X_input)
            y_pred = tf.expand_dims(tf.argmax(y_pred, axis=-1), axis=-1)

        current_input = np.concatenate((current_input[:, steps:, :], y_pred), axis=1)
        predictions.append(y_pred)

    return np.concatenate(predictions, axis=1)


# -----------------------------
# Error Calculation
# -----------------------------
def overall_loss(y_true, y_pred, fout, alpha2pulse_tensor, model_type='lstm', plot_pulses=False):
    """Error between predicted and true centroids."""
    if model_type in ['lstm', 'gru', 'cnn']:
        Y_pred_classes = tf.argmax(y_pred, axis=-1)
        Y_true_classes = tf.squeeze(y_true, axis=-1)
    else:
        Y_pred_classes, Y_true_classes = y_pred, y_true

    decoded_pred = tf.gather(alpha2pulse_tensor, Y_pred_classes)
    decoded_true = tf.gather(alpha2pulse_tensor, Y_true_classes)
    loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.abs(decoded_pred - decoded_true), axis=-1), axis=-1))
    to_log(f"Centroid error: {loss:.3f} | Model: {model_type.upper()}", fout)
    return loss


def overall_loss_2(y_true, Mpulses, test_indices, y_pred, fout, alpha2pulse_tensor,
                   model_type='lstm', plot_pulses=False, mode=0):
    """Error between predicted centroids and true pulses."""
    if model_type in ['lstm', 'gru', 'cnn']:
        Y_pred_classes = tf.argmax(y_pred, axis=-1)
        Y_true_classes = tf.squeeze(y_true, axis=-1)
    else:
        Y_pred_classes, Y_true_classes = y_pred, y_true

    decoded_pred = tf.cast(tf.gather(alpha2pulse_tensor, Y_pred_classes), tf.float32)
    decoded_true_centroids = tf.cast(tf.gather(alpha2pulse_tensor, Y_true_classes), tf.float32)

    decoded_true = []
    for batch_idx, batch_range in enumerate(test_indices):
        decoded_true.append([Mpulses[idx] for idx in batch_range])
    decoded_true = tf.cast(tf.convert_to_tensor(decoded_true), tf.float32)

    loss = tf.reduce_mean(tf.reduce_mean(tf.abs(decoded_pred - decoded_true), axis=-1), axis=-1)
    to_log(f"Pulse vs centroid error: {tf.reduce_mean(loss):.3f} | Model: {model_type.upper()}", fout)
    return tf.reduce_mean(loss)


# -----------------------------
# Visualization Helpers
# -----------------------------
def plot_apds(y_true, y_pred, alpha2pulse_tensor, model_type='lstm'):
    """Plot Action Potential Durations (APDs) for predicted vs true pulses."""
    if model_type in ['lstm', 'gru', 'cnn']:
        Y_pred_classes = tf.argmax(y_pred, axis=-1)
        Y_true_classes = tf.squeeze(y_true, axis=-1)
    else:
        Y_pred_classes, Y_true_classes = y_pred, y_true

    decoded_pred = tf.gather(alpha2pulse_tensor, Y_pred_classes)
    decoded_true = tf.gather(alpha2pulse_tensor, Y_true_classes)

    second_indices_pred = np.argmax(np.cumsum(decoded_pred < 1, axis=2) == 2, axis=2)
    second_indices_true = np.argmax(np.cumsum(decoded_true < 1, axis=2) == 2, axis=2)

    plt.figure(figsize=(16, 4))
    plt.plot(second_indices_pred[0], 'ro-', label='Predicted APDs')
    plt.plot(second_indices_true[0], 'bo-', label='True APDs')
    plt.xlabel('Beat Index')
    plt.ylabel('APD')
    plt.legend()
    plt.show()