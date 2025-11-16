# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

# === Генерация расширенного синтетического датасета ===
def generate_dataset(n=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    data = []
    current_year = 2025
    
    elevator_types = ["пассажирский", "грузовой", "грузопассажирский"]
    type_weights = [0.6, 0.2, 0.2]  # чаще пассажирские
    
    for _ in range(n):
        # Базовые параметры
        inst_year = random.randint(1990, 2020)
        floors = random.randint(5, 25)
        elevator_type = random.choices(elevator_types, weights=type_weights)[0]
        
        # Грузоподъёмность и скорость — зависят от типа
        if elevator_type == "пассажирский":
            capacity = random.choice([400, 630, 800, 1000])
            speed = round(random.uniform(1.0, 2.5), 1)
            daily_uses = 100 + floors * random.randint(20, 30)
        elif elevator_type == "грузовой":
            capacity = random.choice([1000, 1600, 2000])
            speed = round(random.uniform(0.5, 1.0), 1)
            daily_uses = 30 + floors * random.randint(10, 20)
        else:  # грузопассажирский
            capacity = random.choice([800, 1000, 1250])
            speed = round(random.uniform(0.6, 1.6), 1)
            daily_uses = 80 + floors * random.randint(15, 25)
        
        last_repair = random.randint(inst_year, min(current_year, inst_year + 20))
        climate = random.choice([1, 2, 3])  # 1 — тёплый, 3 — холодный
        maintenance_quality = random.randint(1, 5)
        
        # === Имитация осмотров (оценки 1–5) ===
        # Двери (а, д, и)
        base_door = 5 - max(0, (current_year - inst_year - 10) / 5)  # старение
        if elevator_type != "пассажирский":
            base_door -= 0.5  # грузовые — выше износ дверей
        door_score = max(1, min(5, int(np.random.normal(base_door, 0.8))))
        
        # Управление (в, г, к)
        base_control = 5 - (current_year - inst_year) * 0.08
        control_score = max(1, min(5, int(np.random.normal(base_control, 0.7))))
        
        # Безопасность (б, е, л)
        base_safety = 5 - (current_year - last_repair) * 0.2
        safety_score = max(1, min(5, int(np.random.normal(base_safety, 0.6))))
        
        # Общий тренд ухудшения (0 = стабильно, 1 = сильно ухудшилось за год)
        trend = min(1.0, max(0.0, (5 - ((door_score + control_score + safety_score) / 3)) / 5 + random.uniform(-0.1, 0.1)))
        
        min_condition = min(door_score, control_score, safety_score)
        
        # === Расчёт остаточного срока ===
        age = current_year - inst_year
        base_remaining = 25 - age  # норматив 25 лет
        
        # Коррекции
        load_factor = - (daily_uses - 200) / 150
        maint_factor = (maintenance_quality - 3) * 1.2
        climate_factor = -(climate - 2) * 0.8
        repair_factor = (current_year - last_repair) * -0.25
        
        # Влияние состояния узлов
        condition_factor = (door_score + control_score + safety_score - 12) * 0.6  # 12 = 4*3 (среднее 4)
        trend_factor = -trend * 3.0  # сильное ухудшение = большой штраф
        
        remaining = (
            base_remaining +
            load_factor +
            maint_factor +
            climate_factor +
            repair_factor +
            condition_factor +
            trend_factor
        )
        remaining = max(-3, min(20, remaining))
        
        data.append([
            inst_year, floors, elevator_type, capacity, speed, daily_uses,
            last_repair, climate, maintenance_quality,
            door_score, control_score, safety_score, trend, min_condition,
            remaining
        ])
    
    df = pd.DataFrame(data, columns=[
        'installation_year', 'floors', 'elevator_type', 'capacity_kg', 'speed_m_s', 'daily_uses',
        'last_repair_year', 'climate_zone', 'maintenance_quality',
        'avg_door_condition', 'avg_control_condition', 'safety_systems_score', 'overall_condition_trend', 'min_condition_last_year',
        'remaining_life'
    ])
    return df

# === Обучение модели ===
@st.cache_resource
def train_model():
    df = generate_dataset()
    
    # Кодируем категориальный признак
    df = pd.get_dummies(df, columns=['elevator_type'], prefix='type')
    
    # Убедимся, что все ожидаемые столбцы есть
    expected_cols = [
        'installation_year', 'floors', 'capacity_kg', 'speed_m_s', 'daily_uses',
        'last_repair_year', 'climate_zone', 'maintenance_quality',
        'avg_door_condition', 'avg_control_condition', 'safety_systems_score',
        'overall_condition_trend', 'min_condition_last_year',
        'type_грузовой', 'type_грузопассажирский', 'type_пассажирский'
    ]
    
    # Добавим недостающие столбцы нулями (на случай, если какой-то тип не попал в выборку)
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[expected_cols]
    y = df['remaining_life']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, expected_cols, df

model, feature_cols, df = train_model()

# === Streamlit UI ===
st.set_page_config(page_title="ИИ для прогноза срока службы лифтов", layout="centered")
st.title("🏗️ ИИ-система прогнозирования капремонта лифтов")
st.markdown("""
Это приложение оценивает остаточный срок службы лифта на основе:
- технических характеристик,
- условий эксплуатации,
- результатов плановых осмотров (в соответствии с ПБ 10-558-03).
""")

# Ввод данных
st.subheader("🔧 Технические характеристики")
col1, col2 = st.columns(2)

with col1:
    inst_year = st.number_input("Год установки", min_value=1980, max_value=2025, value=2010)
    floors = st.slider("Этажность здания", 5, 25, 9)
    elevator_type = st.selectbox("Тип лифта", ["пассажирский", "грузовой", "грузопассажирский"])

with col2:
    if elevator_type == "пассажирский":
        capacity = st.slider("Грузоподъёмность, кг", 320, 1000, 400, step=50)
        speed = st.slider("Скорость, м/с", 1.0, 2.5, 1.0, step=0.1)
        daily_uses = st.number_input("Поездок в день", 100, 1000, 300)
    elif elevator_type == "грузовой":
        capacity = st.slider("Грузоподъёмность, кг", 1000, 2000, 1000, step=100)
        speed = st.slider("Скорость, м/с", 0.5, 1.0, 0.6, step=0.1)
        daily_uses = st.number_input("Поездок в день", 30, 300, 100)
    else:
        capacity = st.slider("Грузоподъёмность, кг", 800, 1600, 1000, step=50)
        speed = st.slider("Скорость, м/с", 0.6, 1.6, 1.0, step=0.1)
        daily_uses = st.number_input("Поездок в день", 80, 600, 200)

st.subheader("🌡️ Условия эксплуатации")
col3, col4 = st.columns(2)
with col3:
    last_repair = st.number_input("Год последнего капремонта", min_value=1980, max_value=2025, value=2020)
    climate = st.selectbox("Климатический пояс", [(1, "Тёплый"), (2, "Умеренный"), (3, "Холодный")], format_func=lambda x: x[1])[0]
with col4:
    maint_qual = st.slider("Качество техобслуживания (1–5)", 1, 5, 3)

st.subheader("🔍 Результаты последнего осмотра (оценка 1–5)")
col5, col6 = st.columns(2)
with col5:
    door_score = st.slider("Состояние дверей (а, д, и)", 1, 5, 4)
    control_score = st.slider("Состояние управления (в, г, к)", 1, 5, 4)
with col6:
    safety_score = st.slider("Системы безопасности (б, е, л)", 1, 5, 5)
    min_cond = st.slider("Худший показатель за год", 1, 5, min(door_score, control_score, safety_score))

trend = st.slider(
    "Ухудшение состояния за последний год (0 = стабильно, 1 = сильно ухудшилось)",
    0.0, 1.0, 0.2
)

# Прогноз
if st.button("🚀 Рассчитать остаточный срок службы"):
    # Создаём входной вектор
    input_data = pd.DataFrame([{
        'installation_year': inst_year,
        'floors': floors,
        'capacity_kg': capacity,
        'speed_m_s': speed,
        'daily_uses': daily_uses,
        'last_repair_year': last_repair,
        'climate_zone': climate,
        'maintenance_quality': maint_qual,
        'avg_door_condition': door_score,
        'avg_control_condition': control_score,
        'safety_systems_score': safety_score,
        'overall_condition_trend': trend,
        'min_condition_last_year': min_cond,
        'type_грузовой': 1 if elevator_type == "грузовой" else 0,
        'type_грузопассажирский': 1 if elevator_type == "грузопассажирский" else 0,
        'type_пассажирский': 1 if elevator_type == "пассажирский" else 0,
    }])
    
    # Убедимся, что все столбцы на месте
    for col in feature_cols:
        if col not in input_data.columns:
            input_data[col] = 0
    
    input_data = input_data[feature_cols]
    pred = model.predict(input_data)[0]
    
    st.divider()
    if pred < 0:
        st.error(f"⚠️ **Лифт требует немедленной замены!**\n\nЗапас прочности исчерпан на {-pred:.1f} лет.")
        budget = 1_800_000
    else:
        st.success(f"✅ **Ожидаемый остаточный срок службы: {pred:.1f} лет**")
        # Бюджет уменьшается с оставшимся сроком
        budget = max(600_000, int(1_800_000 - pred * 60_000))
        st.info(f"💡 **Рекомендуем заложить в бюджет капремонта: {budget:,} ₽**")

# === Опционально: показать данные ===
with st.expander("📊 Пример синтетических данных (для обучения модели)"):
    st.dataframe(df.head(10))