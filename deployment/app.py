"""
Predicción de Duración de Viajes en Taxi de NYC - Aplicación Streamlit
Carga el pipeline de red neuronal entrenada y proporciona una interfaz interactiva de predicción
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import sys
import importlib
import types
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def _ensure_local_package():
    try:
        importlib.import_module('deployment')
    except ModuleNotFoundError:
        spec = importlib.machinery.ModuleSpec('deployment', loader=None, is_package=True)
        pkg = types.ModuleType('deployment')
        pkg.__path__ = [str(BASE_DIR)]
        pkg.__package__ = 'deployment'
        pkg.__spec__ = spec
        sys.modules.setdefault('deployment', pkg)

_ensure_local_package()

def _alias_module(alias_name, module_name):
    if module_name in sys.modules:
        sys.modules.setdefault(alias_name, sys.modules[module_name])

try:
    from nn_sklearn_wrapper import NeuralNetworkRegressor  # noqa: F401
    from pipeline_components import DummyEncoder, MixedScaler  # noqa: F401
    sys.modules.setdefault('deployment.nn_sklearn_wrapper', sys.modules['nn_sklearn_wrapper'])
    sys.modules.setdefault('deployment.pipeline_components', sys.modules['pipeline_components'])
except ModuleNotFoundError:
    from deployment.nn_sklearn_wrapper import NeuralNetworkRegressor  # noqa: F401
    from deployment.pipeline_components import DummyEncoder, MixedScaler  # noqa: F401
    _alias_module('nn_sklearn_wrapper', 'deployment.nn_sklearn_wrapper')
    _alias_module('pipeline_components', 'deployment.pipeline_components')

# Configuración de página
st.set_page_config(
    page_title="Predictor de Duración de Viajes en Taxi de NYC",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGAR MODELO E INFORMACIÓN
# ============================================================================

@st.cache_resource
def load_pipeline():
    """Cargar el pipeline entrenado desde pickle"""
    pipeline_path = BASE_DIR / 'nn_pipeline.pkl'
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache_data
def load_model_info():
    """Cargar información del modelo"""
    info_path = BASE_DIR / 'model_info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    return info

# Cargar modelo
try:
    pipeline = load_pipeline()
    model_info = load_model_info()
    model_loaded = True
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.info("Asegúrese de que 'nn_pipeline.pkl' y 'model_info.json' estén en el mismo directorio que esta aplicación.")
    model_loaded = False

# ============================================================================
# ENCABEZADO
# ============================================================================

st.markdown('<div class="main-header">Predictor de Duración de Viajes en Taxi de NYC</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predice la duración de viajes en taxi usando una Red Neuronal entrenada en el 70% de los datos de taxis de NYC</div>', unsafe_allow_html=True)

if model_loaded:
    # Mostrar rendimiento del modelo
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("RMSE de Prueba", f"{model_info['test_rmse']:.4f} min")
    with col2:
        st.metric("MAE de Prueba", f"{model_info['test_mae']:.4f} min")
    with col3:
        st.metric("R² de Prueba", f"{model_info['test_r2']:.4f}")

st.markdown("---")

# ============================================================================
# BARRA LATERAL - INFORMACIÓN DEL MODELO
# ============================================================================

if model_loaded:
    with st.sidebar:
        st.header("Información del Modelo")
        
        st.subheader("Arquitectura")
        st.write(f"**Tipo:** {model_info['model_architecture']}")
        st.write(f"**Optimizador:** {model_info['optimizer']}")
        st.write(f"**Tasa de Aprendizaje:** {model_info['learning_rate']}")
        st.write(f"**Activación:** {model_info['activation']}")
        
        st.subheader("Características")
        st.write(f"**Características Numéricas:** {len(model_info['numeric_columns'])}")
        st.write(f"**Características Categóricas:** {len(model_info['categorical_columns'])}")
        
        with st.expander("Ver Detalles de Características"):
            st.write("**Columnas Categóricas:**")
            for col in model_info['categorical_columns']:
                st.write(f"- {col}")
            
            st.write("**Columnas Numéricas:**")
            for col in model_info['numeric_columns'][:10]:  # Mostrar primeras 10
                st.write(f"- {col}")
            if len(model_info['numeric_columns']) > 10:
                st.write(f"... y {len(model_info['numeric_columns']) - 10} más")

# ============================================================================
# INTERFAZ PRINCIPAL
# ============================================================================

if model_loaded:
    # Crear pestañas
    tab1, tab2, tab3 = st.tabs(["Predicción Individual", "Predicción por Lotes", "Acerca de"])
    
    # ========================================================================
    # PESTAÑA 1: PREDICCIÓN INDIVIDUAL
    # ========================================================================
    
    with tab1:
        st.header("Ingrese Detalles del Viaje")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ubicación del Viaje")
            
            pickup_borough = st.selectbox(
                "Distrito de Recogida",
                options=['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island'],
                help="Distrito donde comienza el viaje"
            )
            
            dropoff_borough = st.selectbox(
                "Distrito de Destino",
                options=['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island'],
                help="Distrito donde termina el viaje"
            )
            
        with col2:
            st.subheader("Detalles del Viaje")
            
            distance = st.number_input(
                "Distancia del Viaje (millas)",
                min_value=0.01,
                max_value=100.0,
                value=2.5,
                step=0.1,
                help="Distancia del viaje en millas"
            )
            
            passenger_count = st.number_input(
                "Número de Pasajeros",
                min_value=1,
                max_value=6,
                value=1,
                help="Número de pasajeros"
            )
            
            payment_type = st.selectbox(
                "Tipo de Pago",
                options=['tarjeta de crédito', 'efectivo', 'sin cargo', 'disputa'],
                help="Método de pago"
            )
            
            fare_amount = st.number_input(
                "Monto de Tarifa ($)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=0.5,
                help="Monto base de la tarifa"
            )
            
            extra = st.number_input(
                "Cargos Extra ($)",
                min_value=0.0,
                max_value=50.0,
                value=0.5,
                step=0.25,
                help="Cargos extra (hora pico, nocturno)"
            )
            
            mta_tax = st.number_input(
                "Impuesto MTA ($)",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.25,
                help="Impuesto MTA"
            )
            
            tip_amount = st.number_input(
                "Monto de Propina ($)",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.5,
                help="Monto de propina"
            )
            
            tolls_amount = st.number_input(
                "Monto de Peajes ($)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                help="Cargos de peajes"
            )
        
        st.markdown("---")
        
        # Botón de predicción
        if st.button("Predecir Duración del Viaje", type="primary", use_container_width=True):
            # Crear dataframe de entrada
            input_data = pd.DataFrame({
                'pickup_borough': [pickup_borough],
                'dropoff_borough': [dropoff_borough],
                'payment_type': [payment_type],
                'passenger_count': [passenger_count],
                'trip_distance': [distance],
                'fare_amount': [fare_amount],
                'extra': [extra],
                'mta_tax': [mta_tax],
                'tip_amount': [tip_amount],
                'tolls_amount': [tolls_amount]
            })
            
            # Hacer predicción
            try:
                prediction = pipeline.predict(input_data)[0]
                
                # Mostrar resultado
                st.success("Predicción Completada!")
                
                # Mostrar predicción con formato agradable
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("### Duración del Viaje Predicha")
                    st.markdown(f"<h1 style='text-align: center; color: #FFD700;'>{prediction:.2f} minutos</h1>", 
                               unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; color: #888;'>(~{prediction/60:.1f} horas)</h3>", 
                               unsafe_allow_html=True)
                
                # Información adicional
                st.info(f"""
                **Resumen del Viaje:**
                - Ruta: {pickup_borough} → {dropoff_borough}
                - Distancia: {distance:.2f} millas
                - Velocidad Estimada: {(distance / (prediction/60)):.2f} mph
                - Tarifa Total: ${fare_amount + extra + mta_tax + tip_amount + tolls_amount:.2f}
                """)
                
            except Exception as e:
                st.error(f"Error al hacer la predicción: {e}")
    
    # ========================================================================
    # PESTAÑA 2: PREDICCIÓN POR LOTES
    # ========================================================================
    
    with tab2:
        st.header("Predicción por Lotes desde CSV")
        
        st.info("""
        Suba un archivo CSV con las siguientes columnas:
        - pickup_borough, dropoff_borough, payment_type
        - passenger_count, trip_distance, fare_amount, extra, mta_tax, tip_amount, tolls_amount
        """)
        
        uploaded_file = st.file_uploader("Elija un archivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Cargar datos
                input_df = pd.read_csv(uploaded_file)
                
                st.write(f"Cargadas {len(input_df)} filas")
                st.dataframe(input_df.head())
                
                if st.button("Predecir Todo", type="primary"):
                    # Hacer predicciones
                    with st.spinner("Haciendo predicciones..."):
                        predictions = pipeline.predict(input_df)
                    
                    # Agregar predicciones al dataframe
                    results_df = input_df.copy()
                    results_df['predicted_duration'] = predictions
                    
                    st.success("Predicciones completadas!")
                    
                    # Mostrar estadísticas
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Duración Media", f"{predictions.mean():.2f} min")
                    with col2:
                        st.metric("Duración Mediana", f"{np.median(predictions):.2f} min")
                    with col3:
                        st.metric("Duración Mínima", f"{predictions.min():.2f} min")
                    with col4:
                        st.metric("Duración Máxima", f"{predictions.max():.2f} min")
                    
                    # Mostrar resultados
                    st.dataframe(results_df)
                    
                    # Botón de descarga
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Descargar Predicciones",
                        data=csv,
                        file_name="predicciones.csv",
                        mime="text/csv"
                    )
                    
                    # Visualización
                    fig = px.histogram(
                        predictions,
                        nbins=50,
                        title="Distribución de Duraciones de Viajes Predichas",
                        labels={'value': 'Duración (minutos)', 'count': 'Frecuencia'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error al procesar el archivo: {e}")
    
    # ========================================================================
    # PESTAÑA 3: ACERCA DE
    # ========================================================================
    
    with tab3:
        st.header("Acerca de Este Modelo")
        
        st.markdown("""
        ### Arquitectura de Red Neuronal
        
        Este modelo utiliza una **red neuronal de profundidad media** con las siguientes características:
        
        - **Arquitectura:** Red feedforward de 3 capas con normalización por lotes y dropout
        - **Tamaños de Capas:** 256 → 128 → 64 → 1
        - **Función de Activación:** ReLU
        - **Optimizador:** AdamW con tasa de aprendizaje 0.001
        - **Entrenamiento:** Entrenado en el 70% de los datos de taxis de NYC con parada temprana
        
        ### Rendimiento del Modelo
        
        El modelo fue evaluado en un conjunto de prueba retenido del 30%:
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{model_info['test_rmse']:.4f} minutos", 
                     help="Raíz del Error Cuadrático Medio - más bajo es mejor")
        with col2:
            st.metric("MAE", f"{model_info['test_mae']:.4f} minutos",
                     help="Error Absoluto Medio - más bajo es mejor")
        with col3:
            st.metric("Puntuación R²", f"{model_info['test_r2']:.4f}",
                     help="Coeficiente de determinación - más cercano a 1 es mejor")
        
        st.markdown("""
        ### Por Qué Este Modelo?
        
        Esta red neuronal fue seleccionada como la mejor entre múltiples modelos probados:
        - **Red Neuronal (Ganadora):** Mejor RMSE, MAE y R² en conjunto de prueba
        - **XGBoost:** Segundo cercano, error ligeramente mayor
        - **SVM:** Tercer lugar, error mayor en general
        
        ### Ingeniería de Características
        
        El modelo utiliza características cuidadosamente diseñadas incluyendo:
        - **Ubicación:** Información de distritos (recogida/destino)
        - **Temporal:** Hora, día, día de la semana, mes
        - **Características del Viaje:** Distancia, número de pasajeros
        - **Financiero:** Montos de tarifa, propinas, peajes, impuestos
        
        ### Pipeline de Preprocesamiento
        
        El modelo incluye un pipeline completo de sklearn con:
        1. **Escalado mixto:** StandardScaler para algunas características, PowerTransformer para otras
        2. **Codificación categórica:** Codificación one-hot para distrito y tipo de pago
        3. **Red neuronal:** Modelo PyTorch envuelto en interfaz compatible con sklearn
        
        ### Despliegue del Modelo
        
        El pipeline completo (preprocesamiento + modelo) está serializado como un solo archivo pickle,
        haciendo el despliegue simple y asegurando consistencia entre entrenamiento e inferencia.
        """)
        
        st.markdown("---")
        st.markdown("**Construido con:** PyTorch, scikit-learn, Streamlit")
        st.markdown("**Datos:** Conjunto de Datos de Duración de Viajes en Taxi de NYC")

else:
    st.error("Modelo no cargado. Asegúrese de que todos los archivos requeridos estén presentes.")
