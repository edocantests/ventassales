import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Sales Predictor Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""

.main-header {

font-size: 2.5rem;

color: #1f77b4;

text-align: center;

margin-bottom: 2rem;

}

.metric-card {

background-color: #f0f2f6;

padding: 1rem;

border-radius: 0.5rem;

margin: 0.5rem 0;

}


""", unsafe_allow_html=True)

@st.cache_data
def load_sample_data():
    """Carga datos de ejemplo para la demostración"""
    np.random.seed(42)
    
    # Generar fechas (últimos 2 años)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    data = []
    for date in dates:
        # Simular patrones estacionales
        month = date.month
        day_of_week = date.weekday()
        
        # Efectos estacionales
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
        
        # Efectos semanales
        weekly_factor = 1 + 0.2 * (1 if day_of_week < 5 else 0.8)  # Weekend vs weekday
        
        # Efectos promocionales aleatorios
        promotion_factor = np.random.choice([1.0, 1.5, 2.0], p=[0.7, 0.2, 0.1])
        
        # Base de ventas
        base_sales = 1000
        sales = base_sales * seasonal_factor * weekly_factor * promotion_factor
        sales += np.random.normal(0, 100)  # Ruido
        
        # Características adicionales
        data.append({
            'fecha': date,
            'ventas': max(0, int(sales)),
            'mes': month,
            'dia_semana': day_of_week,
            'es_fin_semana': 1 if day_of_week >= 5 else 0,
            'es_promocion': 1 if promotion_factor > 1.0 else 0,
            'estacion': 'Invierno' if month in [12, 1, 2] else
                       'Primavera' if month in [3, 4, 5] else
                       'Verano' if month in [6, 7, 8] else 'Otoño'
        })
    
    return pd.DataFrame(data)

@st.cache_data
def prepare_features(df):
    """Prepara características para el modelo ML"""
    # Características de fecha
    df['dia'] = df['fecha'].dt.day
    df['dia_año'] = df['fecha'].dt.dayofyear
    df['trimestre'] = df['fecha'].dt.quarter
    
    # Lags (ventas pasadas)
    df = df.sort_values('fecha')
    df['ventas_7d'] = df['ventas'].rolling(7).mean()
    df['ventas_30d'] = df['ventas'].rolling(30).mean()
    
    # Tendencia
    df['tendencia'] = range(len(df))
    
    return df.dropna()

def train_models(X_train, X_test, y_train, y_test):
    """Entrena diferentes modelos ML"""
    models = {}
    results = {}
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    models['Random Forest'] = rf
    results['Random Forest'] = {
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred),
        'predictions': rf_pred
    }
    
    # Regresión Lineal
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    
    models['Regresión Lineal'] = lr
    results['Regresión Lineal'] = {
        'mae': mean_absolute_error(y_test, lr_pred),
        'r2': r2_score(y_test, lr_pred),
        'predictions': lr_pred
    }
    
    return models, results

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Sales Predictor Pro</h1>', unsafe_allow_html=True)
    st.markdown("### **Predictor Inteligente de Ventas para Optimización de Negocio**")
    
    # Sidebar
    st.sidebar.header("🎛️ Configuración")
    
    # Cargar datos
    try:
        df = load_sample_data()
        st.sidebar.success("✅ Datos cargados correctamente")
    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        return
    
    # Selección de modelo
    model_choice = st.sidebar.selectbox(
        "🤖 Modelo de Predicción",
        ["Random Forest", "Regresión Lineal"]
    )
    
    # Configuración de predicción
    st.sidebar.header("📅 Predicción")
    prediction_days = st.sidebar.slider("Días a predecir", 7, 90, 30)
    
    # Preparar datos
    df = prepare_features(df)
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📈 Análisis", "🤖 ML Predictions", "📊 Reportes"])
    
    with tab1:
        st.header("Dashboard de Ventas")
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_ventas = df['ventas'].sum()
            st.metric("💰 Ventas Totales", f"${total_ventas:,}")
        
        with col2:
            avg_ventas = df['ventas'].mean()
            st.metric("📊 Promedio Diario", f"${avg_ventas:,.0f}")
        
        with col3:
            max_ventas = df['ventas'].max()
            st.metric("🚀 Pico de Ventas", f"${max_ventas:,}")
        
        with col4:
            crecimiento = ((df['ventas'].tail(30).mean() - df['ventas'].head(30).mean()) / df['ventas'].head(30).mean()) * 100
            st.metric("📈 Crecimiento", f"{crecimiento:+.1f}%")
        
        # Gráfico de ventas histórico
        st.subheader("📈 Tendencia de Ventas")
        fig = px.line(df, x='fecha', y='ventas', title='Ventas Diarias - Últimos 2 Años')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Análisis Detallado de Ventas")
        
        # Análisis por estación
        st.subheader("🌍 Ventas por Estación")
        estacion_data = df.groupby('estacion')['ventas'].sum().reset_index()
        fig = px.bar(estacion_data, x='estacion', y='ventas', 
                    color='estacion', title='Ventas Totales por Estación')
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis semanal
        st.subheader("📅 Patrones Semanales")
        dia_semana = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        semanal_data = df.groupby('dia_semana')['ventas'].mean().reset_index()
        semanal_data['dia_nombre'] = [dia_semana[i] for i in semanal_data['dia_semana']]
        
        fig = px.bar(semanal_data, x='dia_nombre', y='ventas', 
                    title='Promedio de Ventas por Día de la Semana')
        st.plotly_chart(fig, use_container_width=True)
        
        # Matriz de correlación
        st.subheader("🔍 Análisis de Correlaciones")
        numeric_cols = ['ventas', 'mes', 'dia_semana', 'es_fin_semana', 'es_promocion']
        corr_matrix = df[numeric_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title='Matriz de Correlación')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Predicciones con Machine Learning")
        
        # Preparar datos para ML
        feature_cols = ['mes', 'dia_semana', 'es_fin_semana', 'es_promocion', 
                       'dia', 'dia_año', 'trimestre', 'ventas_7d', 'ventas_30d', 'tendencia']
        X = df[feature_cols]
        y = df['ventas']
        
        # Split train/test
        split_date = df['fecha'].quantile(0.8)
        train_mask = df['fecha'] <= split_date
        test_mask = df['fecha'] > split_date
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # Entrenar modelos
        models, results = train_models(X_train, X_test, y_train, y_test)
        
        # Mostrar métricas del modelo
        st.subheader("📊 Métricas del Modelo")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("MAE (Error Absoluto)", f"{results[model_choice]['mae']:.2f}")
        with col2:
            st.metric("R² Score", f"{results[model_choice]['r2']:.3f}")
        
        # Predicciones vs reales
        st.subheader("🎯 Predicciones vs Valores Reales")
        test_df = df[test_mask].copy()
        test_df['predicciones'] = results[model_choice]['predictions']
        
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Predicciones vs Reales', 'Residuos'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}]])
        
        # Gráfico de dispersión
        fig.add_trace(go.Scatter(x=test_df['ventas'], y=test_df['predicciones'],
                               mode='markers', name='Predicciones',
                               marker=dict(color='blue', size=8)), row=1, col=1)
        
        # Línea de referencia y=x
        min_val = min(test_df['ventas'].min(), test_df['predicciones'].min())
        max_val = max(test_df['ventas'].max(), test_df['predicciones'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                               mode='lines', name='Perfecto', 
                               line=dict(color='red', dash='dash')), row=1, col=1)
        
        # Residuos
        residuos = test_df['ventas'] - test_df['predicciones']
        fig.add_trace(go.Scatter(x=test_df['predicciones'], y=residuos,
                               mode='markers', name='Residuos',
                               marker=dict(color='green', size=8)), row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Importancia de características
        st.subheader("🔍 Importancia de Características")
        if model_choice in models:
            feature_importance = pd.DataFrame({
                'característica': feature_cols,
                'importancia': models[model_choice].feature_importances_
            }).sort_values('importancia', ascending=True)
            
            fig = px.bar(feature_importance, x='importancia', y='característica',
                        orientation='h', title='Importancia de Características')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Reportes y Predicciones Futuras")
        
        # Generar fechas futuras
        last_date = df['fecha'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=prediction_days, freq='D')
        
        # Crear DataFrame futuro
        future_df = pd.DataFrame({
            'fecha': future_dates,
            'mes': [d.month for d in future_dates],
            'dia_semana': [d.weekday() for d in future_dates],
            'es_fin_semana': [1 if d.weekday() >= 5 else 0 for d in future_dates],
            'es_promocion': np.random.choice([0, 1], size=len(future_dates), p=[0.8, 0.2]),
            'dia': [d.day for d in future_dates],
            'dia_año': [d.timetuple().tm_yday for d in future_dates],
            'trimestre': [d.quarter for d in future_dates],
            'tendencia': [len(df) + i for i in range(len(future_dates))]
        })
        
        # Usar últimas ventas para calcular promedios móviles
        for i, row in future_df.iterrows():
            future_df.loc[i, 'ventas_7d'] = df['ventas'].tail(7).mean()
            future_df.loc[i, 'ventas_30d'] = df['ventas'].tail(30).mean()
        
        # Hacer predicciones
        X_future = future_df[feature_cols]
        model = models[model_choice]
        predictions = model.predict(X_future)
        future_df['ventas_predichas'] = np.maximum(0, predictions)  # No ventas negativas
        
        # Mostrar predicciones
        st.subheader(f"🔮 Predicciones Próximos {prediction_days} Días")
        
        col1, col2 = st.columns(2)
        with col1:
            total_prediccion = future_df['ventas_predichas'].sum()
            st.metric("💰 Ventas Predichas", f"${total_prediccion:,.0f}")
        
        with col2:
            avg_prediccion = future_df['ventas_predichas'].mean()
            st.metric("📊 Promedio Diario", f"${avg_prediccion:,.0f}")
        
        # Gráfico de predicciones
        st.subheader("📈 Proyección de Ventas")
        
        # Combinar datos históricos y predicciones
        combined_df = pd.concat([
            df[['fecha', 'ventas']].rename(columns={'ventas': 'ventas_historicas'}),
            future_df[['fecha', 'ventas_predichas']].rename(columns={'ventas_predichas': 'ventas_predichas'})
        ])
        
        fig = go.Figure()
        
        # Datos históricos
        fig.add_trace(go.Scatter(
            x=df['fecha'], y=df['ventas'],
            mode='lines', name='Ventas Históricas',
            line=dict(color='blue')
        ))
        
        # Predicciones
        fig.add_trace(go.Scatter(
            x=future_df['fecha'], y=future_df['ventas_predichas'],
            mode='lines', name='Predicciones',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Proyección de Ventas: Histórico + Predicciones',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabla de predicciones
        st.subheader("📋 Detalle de Predicciones")
        display_df = future_df[['fecha', 'ventas_predichas']].copy()
        display_df['fecha'] = display_df['fecha'].dt.strftime('%Y-%m-%d')
        display_df['ventas_predichas'] = display_df['ventas_predichas'].apply(lambda x: f"${x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True)
        
        # Botón de descarga
        csv = future_df[['fecha', 'ventas_predichas']].to_csv(index=False)
        st.download_button(
            label="📥 Descargar Predicciones (CSV)",
            data=csv,
            file_name=f"predicciones_ventas_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Recomendaciones de negocio
        st.subheader("💡 Recomendaciones de Negocio")
        
        recomendaciones = []
        if future_df['ventas_predichas'].mean() > df['ventas'].mean() * 1.1:
            recomendaciones.append("📈 **Crecimiento esperado**: Considera aumentar el inventario")
        
        if future_df['ventas_predichas'].std() > df['ventas'].std() * 1.2:
            recomendaciones.append("⚠️ **Alta variabilidad**: Implementa estrategias de gestión de riesgo")
        
        fin_semana_promedio = future_df[future_df['es_fin_semana'] == 1]['ventas_predichas'].mean()
        semana_promedio = future_df[future_df['es_fin_semana'] == 0]['ventas_predichas'].mean()
        
        if fin_semana_promedio > semana_promedio * 1.1:
            recomendaciones.append("🏪 **Fines de semana fuertes**: Optimiza horarios y personal de fin de semana")
        
        if not recomendaciones:
            recomendaciones.append("✅ **Predicciones estables**: Mantén estrategia actual")
        
        for rec in recomendaciones:
            st.write(rec)
    
    # Footer
    st.markdown("---")
    st.markdown("**Sales Predictor Pro** - Desarrollado con ❤️ por MiniMax Agent")
    st.markdown("🚀 *Optimiza tu negocio con predicciones inteligentes*")

if __name__ == "__main__":
    main()
