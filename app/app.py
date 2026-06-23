import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from catboost import CatBoostClassifier, Pool
from sklearn.base import BaseEstimator, TransformerMixin
from PIL import Image
import base64
from io import BytesIO
 
# класс препроцессора
class CatBoostPreprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.age_median_ = X.loc[X['age'] != 121, 'age'].median()
        self.tenure_median_ = X.loc[X['tenure_days'] >= 0, 'tenure_days'].median()
        return self
 
    def transform(self, X):
        X = X.copy()
        cols_to_drop = ['campaign', 'type']
        for col in cols_to_drop:
            if col in X.columns:
                X = X.drop(columns=[col])
        X['age'] = X['age'].replace(121, self.age_median_)
        X.loc[X['tenure_days'] < 0, 'tenure_days'] = np.nan
        X['tenure_days'] = X['tenure_days'].fillna(self.tenure_median_)
        X['credit_amount'] = np.log1p(X['credit_amount'])
        X['deposit_amount'] = np.log1p(X['deposit_amount'])
        cat_cols = ['region', 'source', 'credite_active', 'deposit_active', 'month', 'day']
        for col in cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        return X
 
def get_favicon_base64():
    if os.path.exists("2.png"):
        try:
            img = Image.open("2.png")
            # Уменьшаем для favicon (обычно 32x32 или 64x64)
            img = img.resize((64, 64), Image.LANCZOS)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
        except:
            return "📈"  # fallback
    return "📈"  # fallback
 
# настройка страницы
favicon = get_favicon_base64()
st.set_page_config(
    page_title="CRM | Ассистент оператора",
    page_icon=favicon,
    layout="wide"
)
 
# стили
st.markdown("""
<style>
    .stApp { background-color: #f9f9f9; }
    
    [data-testid="stSidebar"] {
        background-color: #1a1a1a !important;
        color: #f0f0f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stSlider label {
        color: #f0f0f0 !important;
    }
    
    .nav-item {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
        width: 100% !important;
        padding: 8px 12px !important;
        border-radius: 8px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        margin-bottom: 2px !important;
        color: #f0f0f0 !important;
        font-size: 15px !important;
        font-weight: 400 !important;
    }
    .nav-item:hover {
        background-color: #333333 !important;
    }
    .nav-item.active {
        background-color: #c62828 !important;
    }
    .nav-item img {
        width: 24px;
        height: 24px;
        filter: brightness(0) invert(1);
        flex-shrink: 0;
        object-fit: contain;
    }
    .nav-item.active img {
        filter: brightness(0) invert(1) !important;
    }
    
    .breadcrumb {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        color: #6c757d;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    .breadcrumb .separator {
        color: #adb5bd;
    }
    .breadcrumb .current {
        color: #2d2d2d;
        font-weight: 500;
    }
    .breadcrumb .parent {
        color: #6c757d;
        cursor: default;
    }
    
    .operator-avatar img {
        width: 80px !important;
        height: 80px !important;
        border-radius: 50% !important;
        object-fit: cover !important;
        border: 2px solid #c62828 !important;
    }
    
    div[data-testid="stMetric"] {
        background-color: white;
        padding: 10px;
        border-radius: 8px;
        border-left: 3px solid #c62828;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetric"] label {
        color: #2d2d2d !important;
        font-weight: 500 !important;
    }
    div[data-testid="stMetric"] .stMetricValue {
        color: #2d2d2d !important;
        font-weight: bold !important;
    }
    div[data-testid="stMetric"] .stMetricDelta {
        color: #c62828 !important;
    }
    
    hr { margin: 1rem 0; border-color: #444; }
    
    div.stButton > button[kind="primary"] {
        background-color: #8b0000 !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.2rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #5c0000 !important;
        color: white !important;
        transform: scale(1.01);
        box-shadow: 0 4px 12px rgba(139,0,0,0.4);
    }
</style>
""", unsafe_allow_html=True)
 
try:
    from gigachat import GigaChat
    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False
 
# загрузка моделей
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('cb_preprocessor.pkl')
        model = CatBoostClassifier()
        model.load_model('catboost_model.cbm')
        calibrator=joblib.load('isotonic_calibrator.pkl')
        return preprocessor, model,calibrator
    except FileNotFoundError:
        st.error("Файлы модели не найдены. Поместите .pkl и .cbm в папку с приложением.")
        st.stop()
 
preprocessor, model, calibrator = load_models()
OPTIMAL_THRESHOLD = 0.14
 
def generate_call_script(client_data, probability):
 
    api_key = st.secrets.get("GIGACHAT_API_KEY")
    if not api_key:
        return "Ключ API GigaChat не найден. Добавьте GIGACHAT_API_KEY в .streamlit/secrets.toml"
 
    
    tenure_years = int(client_data['tenure_days'][0] / 365) if client_data['tenure_days'][0] > 0 else 0
    loyalty = "давний клиент" if tenure_years > 5 else "клиент"
 
    has_credit = str(client_data['credite_active'][0]).lower() in ['active', '1', 'y', 'yes']
    has_deposit = str(client_data['deposit_active'][0]).lower() in ['active', '1', 'y', 'yes']
 
    benefit_hook = ""
    if has_credit:
        benefit_hook = "акцент на том, что бизнес-карта поможет разделить личные и корпоративные расходы, чтобы не путаться в кредитах"
    elif has_deposit:
        benefit_hook = "акцент на интеграции карты с расчетным счетом и начислении процентов на остаток"
    else:
        benefit_hook = "акцент на кэшбэке за бизнес-расходы (реклама, хостинг) и бесплатной онлайн-бухгалтерии"
 
    product_context = """
    Контекст продукта:
    - Кэшбек до 33% у партнеров (Яндекс.Директ, ВК, заправки).
    - Бесплатные переводы на расчетный счет.
    - Доступ в бизнес-залы аэропортов.
    - Встроенная онлайн-бухгалтерия для ИП.
    """
 
    prompt = (
    f"Ты — опытный и естественный оператор колл-центра Альфа-Банка. "
    f"Твоя задача — позвонить клиенту (предпринимателю) и предложить бизнес-карту Альфа-Банка.\n"
    f"{product_context}\n"
    f"Вероятность согласия клиента (только для контекста, НЕ озвучивай): {probability:.0%}\n\n"
    f"Портрет клиента:\n"
    f"- Возраст: {client_data['age'][0]} лет\n"
    f"- Регион: {client_data['region'][0]}\n"
    f"- Срок обслуживания в банке: {tenure_years} лет ({loyalty})\n"
    f"- Главная выгода для клиента: {benefit_hook}\n\n"
    f"Требования:\n"
    f"1. Уважай время предпринимателя. Объем: строго 2-3 предложения. Форма обращения — уважительная.\n"
    f"2. Структура: Имя (придумай) + Приветствие -> Почему звоним: упомяни 1 факт из портрета, например, срок обслуживания в банке -> Оффер (предложение бизнес-карты с конкретной выгодой, связанной с портретом) -> Короткий вопрос на вовлечение (закрытый, чтобы клиенту было легко ответить).\n"
    f"3. НЕ перечисляй данные клиента как анкету. Вплети их естественно в разговор.\n"
    f"4. ЗАПРЕЩЕНО: упоминать вероятность, говорить 'модель предсказала', использовать заезженные маркетинговые клише.\n"
    f"Пиши только текст разговора, без доп. ремарок"

)

 
    try:
       
        with GigaChat(credentials=api_key, verify_ssl_certs=False) as giga:
            response = giga.chat(prompt)
            script = response.choices[0].message.content
            script = script.replace('**', '').replace('##', '').replace('#', '')
            return script
    except Exception as e:
        return f"Ошибка при обращении к Гигачату: {e}"
 
def get_icon_html(icon_name, is_active=False):
    
    icon_file = f"{icon_name}.png"
    if os.path.exists(icon_file):
        try:
            img = Image.open(icon_file)
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            return f'<img src="data:image/png;base64,{img_base64}" style="width:24px;height:24px;filter: brightness(0) invert(1); flex-shrink:0; object-fit:contain;">'
        except:
            return ""
    return ""
 
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Клиенты"
 
# боковая панель
with st.sidebar:
    # Аватарка оператора
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-start; margin-bottom: 0.5rem;" class="operator-avatar">
            <img src="https://leksakov.com/assets/images/resources/132/1litv3991-.jpg" 
                 >
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.text_input("Имя", value="Иванова Н.П.", disabled=True, label_visibility="collapsed")
    st.text_input("Смена", value="Дневная", disabled=True, label_visibility="collapsed")
    
    st.markdown("---")
    
    tabs = ["Мои задачи", "Проекты", "Клиенты", "Отчеты", "Настройки", "Помощь"]
    tab_files = ["tasks", "projects", "clients", "reports", "settings", "help"]
    
    nav_html = ""
    for tab, file in zip(tabs, tab_files):
        is_active = st.session_state.current_tab == tab
        active_class = "active" if is_active else ""
        
        icon_html = get_icon_html(file, is_active)
        
        if icon_html:
            nav_html += f"""
            <div class="nav-item {active_class}" onclick="window.location.href='?tab={tab}'">
                {icon_html}
                <span>{tab}</span>
            </div>
            """
        else:
            nav_html += f"""
            <div class="nav-item {active_class}" onclick="window.location.href='?tab={tab}'">
                <span>{tab}</span>
            </div>
            """
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    query_params = st.query_params
    if 'tab' in query_params:
        st.session_state.current_tab = query_params['tab']
    
    st.markdown("---")
    st.markdown("<h3 style='color:#f0f0f0;'>Настройки модели</h3>", unsafe_allow_html=True)
    threshold = st.slider("Порог вероятности для звонка", 0.10, 0.90, OPTIMAL_THRESHOLD, 0.05)
 
# основная страница
if st.session_state.current_tab == "Клиенты":
    st.markdown("""
    <div class="breadcrumb">
        <span class="parent">Клиенты</span>
        <span class="separator">›</span>
        <span class="current">Анализ клиента</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='color:#2d2d2d;'>Статистика</h3>", unsafe_allow_html=True)
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.metric("Обработано клиентов", "127", delta="+12")
    with col_d2:
        st.metric("Успешных контактов", "89", delta="+8")
    st.caption("За текущую смену")
    st.markdown("---")
    
    st.markdown("<h1 style='color:#2d2d2d; font-weight:500; margin-bottom:0;'>Карточка клиента</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6c757d; margin-top:0;'>Заполните данные и получите рекомендацию по звонку</p>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### Демография")
        age = st.number_input("Возраст", min_value=18, max_value=100, value=40)
        region = st.selectbox("Регион", options=['Москва', 'Санкт-Петербург', 'Другой'], index=2)
        source = st.selectbox("Источник", options=['Интернет', 'Партнер', 'Отделение', 'Другое'], index=2)
    with col2:
        st.markdown("#### Финансы")
        credit_amount = st.number_input("Остаток по кредиту", min_value=0.0, value=50000.0, step=1000.0)
        deposit_amount = st.number_input("Остаток по депозиту", min_value=0.0, value=0.0, step=1000.0)
        credite_active = st.selectbox("Активен по кредитам", options=['N', 'Y'], index=0)
        deposit_active = st.selectbox("Активен по депозитам", options=['N', 'Y'], index=0)
    with col3:
        st.markdown("#### Статус")
        tenure_days = st.number_input("Давность клиента (в днях)", min_value=0, value=1500)
        is_client_new = st.selectbox("Новый клиент", options=[0, 1], index=0)
        is_clientfinalta_active = st.selectbox("Активен по методике финальта", options=[0, 1], index=0)
        was_closed_at_contact = st.selectbox("Закрыт", options=[0, 1], index=0)
 
    month = st.slider("Месяц контакта", 1, 12, 6)
    day = st.slider("День контакта", 1, 31, 15)
 
    st.markdown("---")
    if st.button("Анализировать клиента", type="primary", use_container_width=True):
        input_dict = {
            'age': [age], 'tenure_days': [tenure_days],
            'credit_amount': [credit_amount], 'deposit_amount': [deposit_amount],
            'region': [region], 'source': [source],
            'credite_active': [credite_active], 'deposit_active': [deposit_active],
            'is_client_new': [is_client_new],
            'is_clientfinalta_active': [is_clientfinalta_active],
            'was_closed_at_contact': [was_closed_at_contact],
            'month': [month], 'day': [day]
        }
        input_df = pd.DataFrame(input_dict)
        X_processed = preprocessor.transform(input_df)
        cat_features = ['region', 'source', 'credite_active', 'deposit_active', 'month', 'day']
        for col in cat_features:
            if col in X_processed.columns:
                X_processed[col] = X_processed[col].astype(str)
        eval_pool = Pool(X_processed, cat_features=cat_features)
        raw_proba=model.predict_proba(eval_pool)[:, 1][0]
        proba = calibrator.transform([raw_proba])[0]
        st.session_state['client_data'] = input_dict
        st.session_state['proba'] = proba
        st.session_state['analyzed'] = True
 
    if st.session_state.get('analyzed'):
        proba = st.session_state['proba']
        client_data = st.session_state['client_data']
        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:

            st.markdown("#### Результат скоринга")



            if proba < 0.08:

                st.error("🔴 Низкая вероятность")

                comment = (

                           "Звонок имеет низкую эффективность.")

            elif proba < 0.15:

                st.warning("🟡 Средняя вероятность")

                comment = ( 
                           "Стандартный приоритет обзвона.")

            else:

                st.success("🟢 Высокая вероятность")

                comment = (

                           "Клиент перспективный, рекомендуется приоритетный обзвон.")

           

            st.caption(comment)

           

            st.markdown("---")


            st.markdown(

                f"<div style='color: #888; font-size: 0.85em;'>"

                f"Откалиброванная вероятность: <b>{proba:.1%}</b><br>"

                f"Базовая конверсия по базе: <b>8.0%</b></div>",

                unsafe_allow_html=True

            )

           

            st.markdown("---")

            if proba >= threshold:

                
                st.session_state['should_call'] = True

            else:


                st.session_state['should_call'] = False
         
        with col_res2:
            if st.session_state.get('should_call'):
                st.markdown("#### AI-ассистент")
                st.markdown("Сгенерируйте персонализированный скрипт разговора на основе профиля клиента.")
                if GIGACHAT_AVAILABLE:
                    if st.button("Сгенерировать скрипт разговора", use_container_width=True):
                        with st.spinner('Анализирую профиль...'):
                            script = generate_call_script(client_data, proba)
                            st.markdown("##### Скрипт разговора")
                            st.info(script)
                else:
                    st.warning("Библиотека GigaChat не установлена. Генерация скрипта недоступна.")
            else:
                st.info("Генерация скрипта недоступна — клиент нецелевой (вероятность согласия ниже порога).")
