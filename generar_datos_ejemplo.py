#!/usr/bin/env python3
"""
Generador de Datos de Ejemplo para Sales Predictor Pro
Este script crea diferentes datasets de ejemplo para probar la aplicación
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_retail_data(start_date="2023-01-01", end_date="2024-12-31", seed=42):
    """
    Genera datos realistas de una tienda retail
    """
    np.random.seed(seed)
    random.seed(seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq='D')
    
    data = []
    for date in dates:
        # Factores estacionales realistas
        month = date.month
        day_of_week = date.weekday()
        
        # Estacionalidad por tipo de negocio
        # Enero: rebajas post-navideñas (bajo)
        # Febrero: San Valentín (alto en regalos)
        # Marzo-Mayo: primavera (crecimiento)
        # Junio-Agosto: vacaciones (pico en verano)
        # Septiembre-Octubre: vuelta al cole (alto)
        # Noviembre: Black Friday (muy alto)
        # Diciembre: navidad (muy alto)
        
        seasonal_multipliers = {
            1: 0.8,   # Rebajas
            2: 1.1,   # San Valentín
            3: 1.0,   # Normal
            4: 1.1,   # Primavera
            5: 1.2,   # Buen tiempo
            6: 1.4,   # Vacaciones verano
            7: 1.5,   # Pico verano
            8: 1.3,   # Final vacaciones
            9: 1.4,   # Vuelta al cole
            10: 1.2,  # Otoño
            11: 1.6,  # Black Friday y preparativos
            12: 1.8   # Navidad
        }
        
        # Patrones semanales (retail)
        weekly_patterns = {
            0: 1.0,  # Lunes
            1: 1.1,  # Martes
            2: 1.1,  # Miércoles
            3: 1.2,  # Jueves
            4: 1.3,  # Viernes
            5: 1.5,  # Sábado (pico)
            6: 0.9   # Domingo
        }
        
        # Eventos especiales
        is_black_friday = (month == 11 and date.day in [24, 25, 26, 27])
        is_christmas_season = (month == 12 and date.day >= 15)
        is_summer_sale = (month == 7 and date.day <= 15)
        
        promotion_factor = 1.0
        if is_black_friday:
            promotion_factor = 3.0
        elif is_christmas_season:
            promotion_factor = 2.0
        elif is_summer_sale:
            promotion_factor = 1.5
        elif random.random() < 0.1:  # 10% chance de promoción normal
            promotion_factor = 1.3
        
        # Base de ventas (tienda mediana)
        base_sales = 2500
        sales = (base_sales * 
                seasonal_multipliers[month] * 
                weekly_patterns[day_of_week] * 
                promotion_factor)
        
        # Añadir variación aleatoria
        sales += np.random.normal(0, sales * 0.15)
        sales = max(0, int(sales))
        
        data.append({
            'fecha': date,
            'ventas': sales,
            'mes': month,
            'dia_semana': day_of_week,
            'es_fin_semana': 1 if day_of_week >= 5 else 0,
            'es_promocion': 1 if promotion_factor > 1.2 else 0,
            'es_black_friday': 1 if is_black_friday else 0,
            'es_navidad': 1 if is_christmas_season else 0,
            'trimestre': (month - 1) // 3 + 1,
            'estacion': 'Invierno' if month in [12, 1, 2] else
                       'Primavera' if month in [3, 4, 5] else
                       'Verano' if month in [6, 7, 8] else 'Otoño'
        })
    
    return pd.DataFrame(data)

def generate_restaurant_data(start_date="2023-01-01", end_date="2024-12-31", seed=42):
    """
    Genera datos de un restaurante
    """
    np.random.seed(seed)
    random.seed(seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq='D')
    
    data = []
    for date in dates:
        month = date.month
        day_of_week = date.weekday()
        
        # Restaurantes: picos en fines de semana, estacionalidad verano
        seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * (month - 3) / 12)
        weekly = 1.0 + 0.4 * (1 if day_of_week >= 5 else 0)  # Fines de semana
        
        # Eventos especiales
        is_valentines = (month == 2 and 13 <= date.day <= 15)
        is_mothers_day = (month == 5 and 6 <= date.day <= 8)
        is_new_years = (month == 1 and 1 <= date.day <= 2)
        
        event_factor = 1.0
        if is_valentines:
            event_factor = 2.0
        elif is_mothers_day:
            event_factor = 1.8
        elif is_new_years:
            event_factor = 0.5  # Menos actividad
        
        base_sales = 1200
        sales = base_sales * seasonal * weekly * event_factor
        sales += np.random.normal(0, 200)
        sales = max(0, int(sales))
        
        data.append({
            'fecha': date,
            'ventas': sales,
            'mes': month,
            'dia_semana': day_of_week,
            'es_fin_semana': 1 if day_of_week >= 5 else 0,
            'es_promocion': 1 if random.random() < 0.15 else 0,
            'trimestre': (month - 1) // 3 + 1,
            'estacion': 'Invierno' if month in [12, 1, 2] else
                       'Primavera' if month in [3, 4, 5] else
                       'Verano' if month in [6, 7, 8] else 'Otoño'
        })
    
    return pd.DataFrame(data)

def generate_ecommerce_data(start_date="2023-01-01", end_date="2024-12-31", seed=42):
    """
    Genera datos de e-commerce (más volátiles)
    """
    np.random.seed(seed)
    random.seed(seed)
    
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = pd.date_range(start=start, end=end, freq='D')
    
    data = []
    for date in dates:
        month = date.month
        day_of_week = date.weekday()
        
        # E-commerce: menos estacional pero más volátil
        seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * (month - 11) / 12)
        weekly = 1.0 + 0.3 * (0 if day_of_week in [5, 6] else 1)  # Fines de semana menos activos
        
        # Campañas digitales
        is_double_eleven = (month == 11 and date.day == 11)
        is_double_twelve = (month == 12 and date.day == 12)
        is_cyber_monday = (month == 11 and date.day <= 7)  # Lunes después de Thanksgiving
        
        campaign_factor = 1.0
        if is_double_eleven:
            campaign_factor = 5.0
        elif is_double_twelve:
            campaign_factor = 3.0
        elif is_cyber_monday:
            campaign_factor = 2.0
        elif random.random() < 0.12:  # 12% chance de campaña
            campaign_factor = random.uniform(1.2, 2.0)
        
        base_sales = 3000
        sales = base_sales * seasonal * weekly * campaign_factor
        sales += np.random.normal(0, sales * 0.25)  # Mayor volatilidad
        sales = max(0, int(sales))
        
        data.append({
            'fecha': date,
            'ventas': sales,
            'mes': month,
            'dia_semana': day_of_week,
            'es_fin_semana': 1 if day_of_week >= 5 else 0,
            'es_promocion': 1 if campaign_factor > 1.1 else 0,
            'trimestre': (month - 1) // 3 + 1,
            'estacion': 'Invierno' if month in [12, 1, 2] else
                       'Primavera' if month in [3, 4, 5] else
                       'Verano' if month in [6, 7, 8] else 'Otoño'
        })
    
    return pd.DataFrame(data)

def save_all_datasets():
    """
    Genera y guarda todos los datasets
    """
    print("🚀 Generando datasets de ejemplo...")
    
    # Generar datasets
    retail_df = generate_retail_data()
    restaurant_df = generate_restaurant_data()
    ecommerce_df = generate_ecommerce_data()
    
    # Guardar en CSV
    retail_df.to_csv('ventas_retail_ejemplo.csv', index=False)
    restaurant_df.to_csv('ventas_restaurante_ejemplo.csv', index=False)
    ecommerce_df.to_csv('ventas_ecommerce_ejemplo.csv', index=False)
    
    print("✅ Datasets guardados:")
    print("  📊 ventas_retail_ejemplo.csv - Tienda física")
    print("  🍽️ ventas_restaurante_ejemplo.csv - Restaurante")
    print("  🛒 ventas_ecommerce_ejemplo.csv - Tienda online")
    
    # Mostrar estadísticas
    print("\n📈 Estadísticas de los datasets:")
    print(f"Retail: {retail_df['ventas'].mean():.0f} ± {retail_df['ventas'].std():.0f} (promedio ± std)")
    print(f"Restaurante: {restaurant_df['ventas'].mean():.0f} ± {restaurant_df['ventas'].std():.0f}")
    print(f"E-commerce: {ecommerce_df['ventas'].mean():.0f} ± {ecommerce_df['ventas'].std():.0f}")

if __name__ == "__main__":
    save_all_datasets()
