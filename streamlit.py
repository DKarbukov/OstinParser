import streamlit as st
from geopy.geocoders import Nominatim
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta   
from io import BytesIO
geolocator = Nominatim(user_agent="MyApp")
end_date_default = datetime.now().date()
start_date_default = end_date_default - timedelta(days=7)

start_date = st.sidebar.date_input("Выберите начальную дату",start_date_default)
end_date = st.sidebar.date_input("Выберите конечную дату",end_date_default)

start_date_str = start_date.strftime('%Y/%m/%d')
end_date_str = end_date.strftime('%Y/%m/%d')

# Содержимое страницы "Главная"
def main():
    st.title("Главная страница")
    st.write("Привет! Это главная страница.")
    conn = sqlite3.connect('test.db')
    
    tables = {
        'Комментарии': pd.read_sql_query("SELECT * FROM Comments WHERE Пользователь != '-20367999';", conn),
        'Посты': pd.read_sql_query("SELECT CAST(ID AS TEXT),* FROM Posts;", conn),
        'Комментаторы': pd.read_sql_query("SELECT * FROM Users;", conn),
    }
    # Выбор таблицы для отображения с помощью selectbox
    selected_table = st.selectbox("Выберите таблицу", list(tables.keys()))
    if selected_table == 'Комментарии':
        # post_filter_value = st.text_input("Введите ID постов (разделите их запятой)", key='post_ids_input')
        # post_ids = [int(post_id.strip()) for post_id in post_ids_input.split(', ') if post_id.strip()] if post_ids_input else []

        # if post_ids:
        #     query_comments = f"SELECT Пост, Sentiment FROM Comments WHERE Пост IN ({', '.join(map(str, post_ids))})"
        #     filtered_data = pd.read_sql_query(query_comments, conn)
        sentiment_filter = st.sidebar.multiselect("Фильтр по тональности", tables[selected_table]['Sentiment'].unique(), default=[])
        country_filter = st.sidebar.multiselect("Фильтр по стране", tables[selected_table]['Country'].unique(), default=[])
        city_filter = st.sidebar.multiselect("Фильтр по городу", tables[selected_table]['City'].unique(), default=[])
        sex_filter = st.sidebar.multiselect("Фильтр по полу", tables[selected_table]['Sex'].unique(), default=[])
        # post_filter_value = st.text_area("Поиск всех комментариев к определенным постам (столбец 'Пост')")
       
        # date_type = st.radio("Выберите тип фильтрации по дате", ["Диапазон дат", "Конкретная дата", 'Все'], index=2)
        # if date_type == "Диапазон дат":
        #     start_date_tab = st.date_input("Выберите начальную дату", datetime.now() - timedelta(days=7), key='start_date_tab')
        #     end_date_tab = st.date_input("Выберите конечную дату", datetime.now(),key='end_date_tab')
        #     start_date_tab_str = start_date_tab.strftime('%Y/%m/%d')
        #     end_date_tab_str = end_date_tab.strftime('%Y/%m/%d')
        #     filtered_data = tables[selected_table][(tables[selected_table]['Дата'] >= start_date_tab_str) & (tables[selected_table]['Дата'] <= end_date_tab_str)]
        # elif date_type == "Конкретная дата":
        #     selected_date = st.date_input("Выберите конкретную дату", datetime.now())
        #     selected_date_str = selected_date.strftime('%Y/%m/%d')
        #     filtered_data = tables[selected_table][tables[selected_table]['Дата'] == selected_date_str]
        # else: 
        #     filtered_data = tables[selected_table]
        filtered_data = tables[selected_table][(tables[selected_table]['Дата'] >= start_date_str) & (tables[selected_table]['Дата'] <= end_date_str)]

        if sentiment_filter:
            filtered_data = filtered_data[filtered_data['Sentiment'].isin(sentiment_filter)]
        if city_filter:
            filtered_data = filtered_data[filtered_data['City'].isin(city_filter)]
        if country_filter:
            filtered_data = filtered_data[filtered_data['Country'].isin(country_filter)]
        if sex_filter:
            filtered_data = filtered_data[filtered_data['Sex'].isin(sex_filter)]
        
        # post_filter_value = st.number_input("Введите ID постов (столбец 'Пост') (разделите их запятой)",min_value=0)
        # if post_filter_value:
        #     filtered_data = filtered_data[filtered_data['Пост'] == post_filter_value]
        
        post_filter_value = st.text_input("Поиск всех комментариев к определенным постам (столбец 'Пост')", help ='Разделите запятой и пробелом')
        if post_filter_value:
            # Разделение введенных значений по переносу строки или другому разделителю
            post_filter_values_list = [x.strip() for x in post_filter_value.split(', ') if x.strip()]
            # Применение фильтрации по столбцу "Пост" для всех введенных значений
            filtered_data = filtered_data[filtered_data['Пост'].astype(str).isin(post_filter_values_list)]
        
        id_filter_value = st.number_input("Поиск всех комментариев человека по его ID (столбец 'Пользователь')", min_value=0, help ='Разделите запятой и пробелом')
        if id_filter_value:
            # Применение фильтрации по столбцу ID
            filtered_data = filtered_data[filtered_data['Пользователь'] == id_filter_value]

        filtered_data['ID'] = filtered_data['ID'].astype(str)
        filtered_data['Пост'] = filtered_data['Пост'].astype(str)
        filtered_data['Пользователь'] = filtered_data['Пользователь'].astype(str)

        filtered_data = filtered_data.reset_index(drop=True)
        
        st.write(filtered_data)
        
        if st.button("Выгрузить в Excel", key = 'button1'):
            excel_file = BytesIO()
            filtered_data.to_excel(excel_file, index=False, engine='openpyxl')
            excel_file.seek(0)
            st.download_button(
                label="Скачать Excel файл",
                data=excel_file,
                file_name=f"{selected_table}.xlsx",
                key="download_button1",
            )
        if not filtered_data.empty:
            color_map = {'Positive': '#00FF7F', 'Negative': '#E32636'}

            
            fig_pie = px.pie(filtered_data, names='Sentiment', title='Анализ тональности комментариев', color_discrete_map=color_map)

            st.plotly_chart(fig_pie)
        else:
            st.write("Нет данных для отображения.")

    if selected_table == "Посты":
        # Добавляем фильтр по значению столбца "Sentiment"
        sentiment_filter = st.sidebar.multiselect("Фильтр по тональности", tables[selected_table]['Sentiment'].unique(), default=[])
        # date_type = st.radio("Выберите тип фильтрации по дате", ["Диапазон дат", "Конкретная дата", 'Все'], index=2)
        # if date_type == "Диапазон дат":
        #     start_date_tab = st.date_input("Выберите начальную дату", datetime.now() - timedelta(days=7), key='start_date_tab1')
        #     end_date_tab = st.date_input("Выберите конечную дату", datetime.now(),key='end_date_tab1')
        #     start_date_tab_str = start_date_tab.strftime('%Y/%m/%d')
        #     end_date_tab_str = end_date_tab.strftime('%Y/%m/%d')
        #     filtered_data = tables[selected_table][(tables[selected_table]['Дата'] >= start_date_tab_str) & (tables[selected_table]['Дата'] <= end_date_tab_str)]
        # elif date_type == "Конкретная дата":
        #     selected_date = st.date_input("Выберите конкретную дату", datetime.now())
        #     selected_date_str = selected_date.strftime('%Y/%m/%d')
        #     filtered_data = tables[selected_table][tables[selected_table]['Дата'] == selected_date_str]
        # else: 
        #     filtered_data = tables[selected_table]
        filtered_data = tables[selected_table][(tables[selected_table]['Date'] >= start_date_str) & (tables[selected_table]['Date'] <= end_date_str)]
    
        
        if sentiment_filter:
            filtered_data = filtered_data[filtered_data['Sentiment'].isin(sentiment_filter)]
        else:
            # Если нет фильтра, отображаем все строки
            filtered_data = tables[selected_table][(tables[selected_table]['Date'] >= start_date_str) & (tables[selected_table]['Date'] <= end_date_str)]
         # Сбрасываем индекс и устанавливаем начальное значение в 1
        filtered_data = filtered_data.reset_index(drop=True)
        
        filtered_data['ID'] = filtered_data['ID'].astype(str)
        filtered_data['Views'] = filtered_data['Views'].astype(str)

        st.write(filtered_data)
        if st.button("Выгрузить в Excel", key = 'button2'):
            excel_file = BytesIO()
            filtered_data.to_excel(excel_file, index=False, engine='openpyxl')
            excel_file.seek(0)
            st.download_button(
                label="Скачать Excel файл",
                data=excel_file,
                file_name=f"{selected_table}.xlsx",
                key="download_button2",
            )
        filtered_data['Sentiment'].fillna('Neutral', inplace=True)
        #График, который берет значения столбца "Sentiment" и показывает на круговой диаграмме % появления этих значений 
        if not filtered_data.empty:
            color_map = {'Positive': '#00FF7F', 'Negative': '#E32636'}
            fig_pie = px.pie(filtered_data, names='Sentiment', title='Анализ общей тональности комментариев на постах', color_discrete_map=color_map)

            st.plotly_chart(fig_pie)
        else:
            st.write("Нет данных для отображения.")
    
    if selected_table == 'Комментаторы':
        country_filter = st.sidebar.multiselect("Фильтр по стране", tables[selected_table]['Country'].unique(), default=[],key = 'user_country')
        city_filter = st.sidebar.multiselect("Фильтр по городу", tables[selected_table]['City'].unique(), default=[],key = 'user_city')
        sex_filter = st.sidebar.multiselect("Фильтр по полу", tables[selected_table]['Sex'].unique(), default=[],key = 'user_sex')
        if st.button('Показать карту'):
            st.write('Карта загружается...')
            cities = tables[selected_table]['City'].unique()
            conn2 = sqlite3.connect('Coordinates.db')
            exist_cities = pd.read_sql_query("SELECT * FROM Cities;", conn2)
            only_names = exist_cities['Город'].values.tolist()
            cities_coord = {'lat':[], 'lon': [], 'Город': []}
            # print(only_names)
            for i in cities:
                if i not in only_names and i != None:
                    # print(i)
                    coord = geolocator.geocode(i)
                    if coord is not None:
                        cities_coord['lat'].append(coord.latitude)
                        cities_coord['lon'].append(coord.longitude)
                        cities_coord['Город'].append(i)
                elif i in only_names and i != None:
                    # print(f'{i} exists')
                    cities_coord['lat'].append(exist_cities[exist_cities['Город']==i]['lat'].values[0])
                    cities_coord['lon'].append(exist_cities[exist_cities['Город']==i]['lon'].values[0])
                    cities_coord['Город'].append(i)
            final_coord = pd.DataFrame(cities_coord)
            # print(cities_coord)
            final_coord.to_sql('Cities', conn2, if_exists='append', index=False)
            st.write('Карта готова!')
            st.map(final_coord)

        
        filtered_data = tables[selected_table]

        if city_filter:
            filtered_data = filtered_data[filtered_data['City'].isin(city_filter)]
        if country_filter:
            filtered_data = filtered_data[filtered_data['Country'].isin(country_filter)]
        if sex_filter:
            filtered_data = filtered_data[filtered_data['Sex'].isin(sex_filter)]

        filtered_data = filtered_data.reset_index(drop=True)
        
        filtered_data['ID'] = filtered_data['ID'].astype(str)
        st.write(filtered_data)

    
        if st.button("Выгрузить в Excel", key = 'button3'):
            excel_file = BytesIO()
            filtered_data.to_excel(excel_file, index=False, engine='openpyxl')
            excel_file.seek(0)
            st.download_button(
                label="Скачать Excel файл",
                data=excel_file,
                file_name=f"{selected_table}.xlsx",
                key="download_button3",
            )

        if not filtered_data.empty:
            # Разделение на две колонки
            col1, col2 = st.columns(2)
            
        
            with col1:
                color_map_sex = {'Male': '#FF69B4', 'Female': '#1f77b4'}
                fig_pie_sex = px.pie(filtered_data, names='Sex', title='Пол комментаторов', color_discrete_map=color_map_sex)
                fig_pie_sex.update_layout(legend=dict(orientation='h'))
                st.plotly_chart(fig_pie_sex, use_container_width=True, height=400)
            # График круговой диаграммы для 'Subscriber'
            with col2:
                # Подсчет частоты значений '1' и '0' в столбце 'Subscriber'
                subscriber_count = filtered_data['Subscriber'].value_counts()
                subscriber_count.index = subscriber_count.index.map({1: 'Комментаторы, которые подписаны', 0: 'Комментаторы, которые не подписаны'})
                # Создание круговой диаграммы
                fig_pie_subscriber = px.pie(subscriber_count, values=subscriber_count.values, names=subscriber_count.index, 
                                            title='Анализ подписчиков')
                fig_pie_subscriber.update_traces(textinfo='percent')
                fig_pie_subscriber.update_layout(legend=dict(orientation='h'))
                st.plotly_chart(fig_pie_subscriber, width=200, height=400)
        else:
            st.write("Нет данных для отображения.")

def statistics():
    
    conn = sqlite3.connect('test.db')
    st.header('Динамика показателей')


    st.subheader('Динамика просмотров')
    query_views = f"SELECT Date, Views, ID FROM Posts WHERE Date BETWEEN '{start_date_str}' AND '{end_date_str}';"
    data_views = pd.read_sql_query(query_views, conn)


    if not data_views.empty:
        data_views['ID'] = data_views['ID'].apply(lambda x: f'<a href="https://vk.com/ostin?w=wall-20367999_{x}" target="_blank">{x}</a>')

        fig = px.line(data_views, x='ID', y='Views', title='Динамика просмотров в выбранном временном периоде')

        fig.update_layout(
            xaxis=dict(title='ID постов', type='category'),
            yaxis=dict(title='Просмотры'),
            title='Динамика просмотров постов в выбранном временном периоде'
        )

        st.plotly_chart(fig)
    else:
        st.write("Нет данных для отображения.")



    st.subheader('Общий анализ тональности комментариев')
    
    post_ids_input = st.text_input("Введите ID постов (разделите их запятой)", key='post_ids_input')
    post_ids = [int(post_id.strip()) for post_id in post_ids_input.split(', ') if post_id.strip()] if post_ids_input else []

    if post_ids:
        query_comments = f"SELECT Пост, Sentiment FROM Comments WHERE Пост IN ({', '.join(map(str, post_ids))})"
    else: 
        query_comments = f"SELECT Пост, Sentiment FROM Comments WHERE Пост IN (SELECT ID FROM Posts WHERE Date BETWEEN '{start_date_str}' AND '{end_date_str}');"
    data_comments = pd.read_sql_query(query_comments, conn)


    if not data_comments.empty:
        color_map = {'Positive': '#00FF7F','Negative': '#E32636'}
        fig_pie = px.pie(data_comments, names='Sentiment', title='Анализ тональности комментариев', color_discrete_map = color_map)

        st.plotly_chart(fig_pie)
    else:
        st.write("Нет данных для отображения.")


    st.subheader('Активность на постах')
    if post_ids:
        query_posts = f"SELECT ID, Comments, Likes, Reposts FROM Posts WHERE ID IN ({', '.join(map(str, post_ids))})"
    else:
        query_posts = f"SELECT ID, Comments, Likes, Reposts FROM Posts WHERE Date BETWEEN '{start_date_str}' AND '{end_date_str}'"
    
    data_posts = pd.read_sql_query(query_posts, conn)

    data_posts['ID'] = data_posts['ID'].apply(lambda x: f'<a href="https://vk.com/ostin?w=wall-20367999_{x}" target="_blank">{x}</a>')

    if not data_posts.empty:

        fig = px.bar(data_posts, x='ID', y=['Comments', 'Likes', 'Reposts'], barmode='group')

        
        fig.update_layout(
            xaxis=dict(title='ID'),
            yaxis=dict(title='Значения'),
            title='Анализ реакций на посты'
        )

        st.plotly_chart(fig)
    else:
        st.write("Нет данных для отображения.")

    st.markdown("")
    st.markdown("")


    st.subheader('Тональность отзывов')
    if post_ids:
        query_comments = f"SELECT Пост, Sentiment FROM Comments WHERE Пост IN ({', '.join(map(str, post_ids))})"
    else:
        query_comments = f"SELECT Пост, Sentiment FROM Comments WHERE Пост IN (SELECT ID FROM Posts WHERE Date BETWEEN '{start_date_str}' AND '{end_date_str}');"
    data_comments = pd.read_sql_query(query_comments, conn)

    if not data_comments.empty:

        grouped = data_comments.groupby(['Пост', 'Sentiment']).size().reset_index(name='Частота')
        grouped['Пост'] = grouped['Пост'].apply(lambda x: f'<a href="https://vk.com/ostin?w=wall-20367999_{x}" target="_blank">{x}</a>')
    
        color_map = {'Positive': 'lightgreen'}
        fig = px.bar(grouped, x='Пост', y='Частота', color='Sentiment', 
                    title='Анализ тональности комментариев',
                    labels={'Посты': 'Посты', 'Частота': 'Количество'}, color_discrete_map=color_map)

        # Настройка осей и подписей
        fig.update_layout(xaxis_type='category', xaxis_title='Посты', yaxis_title='Количество')

        st.plotly_chart(fig)
    else:
        st.write("Нет данных для отображения.")

    conn.close()

def tops():
    conn = sqlite3.connect('test.db')
    
    st.title('Топ постов')
    filter_options = ['Comments', 'Likes', 'Views', 'Reposts']
    selected_filter = st.selectbox('Выберите фильтр для сортировки', filter_options)
    
    num_to_display = st.number_input("Введите количество постов и комментариев для отображения", min_value=1, value=10)
    
    st.subheader(f"**Топ {num_to_display} постов, отсортированных по {selected_filter}:**")

    query = f"SELECT * FROM Posts WHERE Date BETWEEN '{start_date_str}' AND '{end_date_str}' ORDER BY {selected_filter} DESC LIMIT {num_to_display};"
    data = pd.read_sql_query(query, conn)
    
    if not data.empty:
        data.index += 1  
        if 'ID' in data.columns:
            data['ID'] = data['ID'].apply(lambda x: f'<a href="https://vk.com/ostin?w=wall-20367999_{x}">{x}</a>')
        
        st.write(data.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.write("Нет данных для отображения.")

    if st.button("Выгрузить в Excel", key = 'button3'):
        excel_file = BytesIO()
        data.to_excel(excel_file, index=False, engine='openpyxl')
        excel_file.seek(0)
        st.download_button(
            label="Скачать Excel файл",
            data=excel_file,
            file_name="Топ постов.xlsx",
            key="download_button3",
        )
    st.markdown("")
    st.markdown("")
    st.subheader(f"Топ {num_to_display} комментариев, отсортированных по Лайкам:")
    
    
    query_comments = f"SELECT ID, Дата, Комментарий, Лайки, Sentiment FROM Comments WHERE Дата BETWEEN '{start_date_str}' AND '{end_date_str}' ORDER BY Лайки DESC LIMIT {num_to_display};"
    data_comments = pd.read_sql_query(query_comments, conn)


    if not data_comments.empty:
        
        data_comments.index += 1
        data_comments['ID'] = data_comments['ID'].apply(lambda x: f'<a href="https://vk.com/ostin?w=wall-20367999_{x}">{x}</a>')
        st.write(data_comments.to_html(escape=False), unsafe_allow_html=True)
    else:
        st.write("Нет данных для отображения в таблице Comments.")
    if st.button("Выгрузить в Excel", key = 'button4'):
        excel_file = BytesIO()
        data_comments.to_excel(excel_file, index=False, engine='openpyxl')
        excel_file.seek(0)
        st.download_button(
            label="Скачать Excel файл",
            data=excel_file,
            file_name="Топ комментариев.xlsx",
            key="download_button4",
        )
    
    conn.close()

# def audience():
    # st.header('Анализ аудитории')
    # conn = sqlite3.connect('test.db')
    # query_users = "SELECT Sex FROM Users;"
    # data_users = pd.read_sql_query(query_users, conn)

    # if not data_users.empty:
    #     fig_pie = px.pie(data_users, names='Sex', title='Пол комментаторов')
    #     fig_pie.update_traces(marker=dict(colors=['#FF69B4', '#1f77b4']))
    #     st.plotly_chart(fig_pie)
    # else:
    #     st.write("Нет данных для отображения.")
   
    # # Загрузка данных из таблицы Users
    # query = "SELECT Subscriber FROM Users;"
    # data = pd.read_sql_query(query, conn)

    # # Подсчет частоты значений '1' и '0' в столбце 'Subscriber'
    # subscriber_count = data['Subscriber'].value_counts()

    # subscriber_count.index = subscriber_count.index.map({1: 'Комментаторы, которые подписаны', 0: 'Комментаторы, которые не подписаны'})

    # # Создание круговой диаграммы
    # fig = px.pie(subscriber_count, values=subscriber_count.values, names=subscriber_count.index, 
    #             title='Анализ подписчиков')

    # # Настройка подписей
    # fig.update_traces(textinfo='percent')

    # # Отображение диаграммы
    # st.plotly_chart(fig)

    # conn.close()

# Опции для навигации между страницами
pages = {
    "Главная": main,
    "Анализ статистики": statistics,
    'Топ': tops,
    # 'Анализ аудитории': audience
}

# Боковая панель для выбора страниц
selection = st.sidebar.radio("Выберите страницу", list(pages.keys()))

# Отображение выбранной страницы
pages[selection]()
