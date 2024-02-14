from APIVK_private import MY_TOKEN
from emosent import get_emoji_sentiment_rank_multiple
import vk_api
import datetime
import pandas as pd
import sqlite3
import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
login = vk_api.VkApi(token=MY_TOKEN)
ostin_id = -20367999


def get_group_posts(group_id, count):   #Collects posts and separates the ones without comments
    offset = 0
    posts_with_comments, all_posts = {}, {}
    while offset < count:
        if count - offset > 100: count_lim = 100
        else: count_lim = count - offset
        posts = login.method("wall.get", {"owner_id": group_id, "count": count_lim, "offset": offset})
        offset += count_lim
        for i in posts['items']:
            all_posts[len(all_posts)+1] = {'ID': i['id'], 'Date': datetime.datetime.utcfromtimestamp(i['date']).strftime('%Y/%m/%d'), 'Comments': i['comments']['count'], 'Likes': i['likes']['count'], 'Views': i['views']['count'], 'Reposts': i['reposts']['count']}
            if i['comments']['count'] > 0:
                posts_with_comments[len(posts_with_comments)+1] = {'ID': i['id'], 'Date': datetime.datetime.utcfromtimestamp(i['date']).strftime('%Y/%m/%d'), 'Comments': i['comments']['count'], 'Likes': i['likes']['count'], 'Views': i['views']['count'], 'Reposts': i['reposts']['count']}
        print(f'Collected {offset}')
    print('Finished scanning posts')
    return all_posts, posts_with_comments


def get_comments(group_id, posts):  #Gets comments, gets answers to these comments, assigns sentiment value, changes post rating
    comments_list_by_post = []
    clean_comments = {}
    names_ids = []
    clean_names = {}
    for i in posts:
        comments_list_by_post.append(login.method("wall.getComments", {"owner_id": group_id, "post_id": posts[i]['ID'], "need_likes": 1}))
    print(f'Collected comments from {len(comments_list_by_post)} posts')
    print('Starting to sort comments')
    for i in comments_list_by_post:
        for a in i['items']:
            try:
                clean_comments[len(clean_comments)] = {'ID': a['id'], 'Пост': a['post_id'],
                                                        'Пользователь': a['from_id'],
                                                        'Комментарий': a['text'],
                                                        'Лайки': a['likes']['count'],
                                                        'Ответы': a['thread']['count'],
                                                        'Дата': datetime.datetime.utcfromtimestamp(a['date']).strftime('%Y/%m/%d')}
                names_ids.append(a['from_id'])
            except Exception as ex:
                clean_comments[len(clean_comments)] = {'ID': '', 'Пост': ex,
                                                        'Пользователь': '',
                                                        'Комментарий': '',
                                                        'Лайки': '',
                                                        'Ответы': '',
                                                        'Дата': ''}
            if a['thread']['count'] > 0:
                thread = login.method("wall.getComments", {"owner_id": group_id, "comment_id": a['id'], "need_likes": 1})
                for answer in thread['items']:
                    try:
                        clean_comments[len(clean_comments)] = {'ID': a['id'], 'Пост': answer['post_id'],
                                                                'Пользователь': answer['from_id'],
                                                                'Комментарий': answer['text'],
                                                                'Лайки': answer['likes']['count'],
                                                                'Ответы': f'Ответ {a["from_id"]}',
                                                                'Дата': datetime.datetime.utcfromtimestamp(answer['date']).strftime('%Y/%m/%d')}
                        names_ids.append(answer['from_id'])
                    except Exception as ex:
                        clean_comments[len(clean_comments)] = {'ID': '', 'Пост': ex,
                                                                'Пользователь': '',
                                                                'Комментарий': '',
                                                                'Лайки': '',
                                                                'Ответы': '',
                                                                'Дата': ''}
    for key, comment in clean_comments.items():
        text = comment['Комментарий']
        if get_emoji_sentiment_rank_multiple(text) == []:
            sentiment, certainty = predict(text)

            # if sentiment == "Positive":
            #     clean_comments[key]['Rating'] = 1 + clean_comments[key]['Лайки']
            #     for i in posts:
            #         if posts[i]['ID'] == clean_comments[key]['Пост']:
            #             posts[i]['Rating'] = posts[i]['Rating'] + clean_comments[key]['Rating']
            #             break
            # elif sentiment == "Negative":
            #     clean_comments[key]['Rating'] = -1 - clean_comments[key]['Лайки']
            #     for i in posts:
            #         if posts[i]['ID'] == clean_comments[key]['Пост']:
            #             posts[i]['Rating'] = posts[i]['Rating'] + clean_comments[key]['Rating']
            #             break 
        else:
            pos_value, neg_value, neut_value = 0, 0, 0
            temp = get_emoji_sentiment_rank_multiple(text)
            for i in temp:
                pos_value += i['emoji_sentiment_rank']['positive']
                neg_value += i['emoji_sentiment_rank']['negative']
                neut_value += i['emoji_sentiment_rank']['neutral']
            certainty = '-'
            if pos_value > neg_value and pos_value > neut_value: sentiment = 'Positive'
            elif neg_value > pos_value and neg_value > neut_value: sentiment = 'Negative'
            else: sentiment = 'Neutral'
        clean_comments[key]['Sentiment'] = sentiment if sentiment else 'Not analyzed' 
        clean_comments[key]['Certainty'] = certainty if sentiment else 'Not analyzed'
        print(f'Sentiment analysis {key}/{len(clean_comments.items())}') 
    print(f'Starting to retrieve {len(names_ids)} names')
    names = login.method("users.get", {"user_ids": str(names_ids)[1:-1], "fields": "city, country, sex", "name_case": "nom"})
    names_no_ostin = [value for value in names_ids if value > 0]
    sub_status = login.method("groups.isMember", {"group_id": int(str(group_id)[1:]), "user_ids": str(names_no_ostin)[1:-1]})
    SEX = {1: 'Female', 2: 'Male'}
    for i in names:
        clean_names[len(clean_names)+1] = {'ID': i.get('id'), 'First name': i.get('first_name'), 'Last name': i.get('last_name'), 'Subscriber':sub_status[len(clean_names)]["member"], 'City': i.get('city'), 'Country': i.get('country'), 'Sex': SEX[i.get('sex')]}
    print('Generating df')

    return clean_comments, clean_names

def predict(text):  #Gets sentiment value
    tokenizer = BertTokenizerFast.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment')
    model = AutoModelForSequenceClassification.from_pretrained('blanchefort/rubert-base-cased-sentiment-rusentiment', return_dict=True)
    CASES = {0: "Neutral", 1: "Positive", 2: "Negative"}
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    result = CASES[int(torch.argmax(predicted).numpy())]
    certainty = f'{round(torch.max(predicted).item() * 100, 2)}%'
    return result, certainty


def export_to_db(spisok, table):
    if table == 'Users':
        df = pd.DataFrame(spisok).transpose().reset_index(drop=True)
        df['City'] = df['City'].apply(lambda x: x['title'] if x is not None else None)
        df['Country'] = df['Country'].apply(lambda x: x['title'] if x is not None else None)
    else:
        df = pd.DataFrame(spisok)
        df = df.transpose()
        df = df.reset_index(drop=True)
    db_file = 'test.db'
    conn = sqlite3.connect(db_file)
    try:
        if table == 'Comments':
            cursor = conn.cursor()
            for index, row in df.iterrows():
                id = row['ID']
                post_id = row['Пост']
                user = row['Пользователь']
                comments_for_table = row['Комментарий']
                date_for_table = row['Дата']
                likes = row['Лайки']
                sentiment = row['Sentiment']
                certainty = row['Certainty']
                answers = row['Ответы']
                cursor.execute(f"SELECT * FROM {table} WHERE ID = ?", (id,))
                existing_row = cursor.fetchone()
                if existing_row and str(answers).isnumeric():
                    # If the post_id exists, update likes and reposts
                    cursor.execute(f"UPDATE {table} SET 'Лайки' = ?, Sentiment = ? WHERE ID = ?",
                                (likes, sentiment, id))
                elif not existing_row:   
                    cursor.execute(f"INSERT INTO {table} (ID, 'Пост', 'Пользователь', 'Комментарий', 'Лайки', 'Ответы', 'Дата', Sentiment, Certainty) VALUES ( ?, ?, ?, ?, ?, ?, ?, ?, ? )",
                                (id, post_id, user, comments_for_table, likes, answers, date_for_table, sentiment, certainty))
        elif table == 'Posts':
            cursor = conn.cursor()
            for index, row in df.iterrows():
                post_id = row['ID']
                likes = row['Likes']
                date = row ['Date']
                views = row['Views']
                reposts = row['Reposts']
                comments_for_table = row['Comments']
                cursor.execute(f"SELECT * FROM {table} WHERE ID = ?", (post_id,))
                existing_row = cursor.fetchone()
                if existing_row:
                    cursor.execute(f"UPDATE {table} SET Likes = ?, Views = ?, Reposts = ?, Comments = ? WHERE ID = ?",
                                (likes, views, reposts, comments_for_table, post_id))
                else:
                    cursor.execute(f"INSERT INTO {table} (ID, Date, Comments, Likes, Views, Reposts) VALUES ( ?, ?, ?, ?, ?, ? )",
                                (post_id, date, comments_for_table, likes, views, reposts))
        elif table == 'Users':
            existing_data = pd.read_sql_query(f'SELECT * FROM {table}', conn)
            existing_data = existing_data.reset_index(drop=True)
            new_data = df[~df['ID'].isin(existing_data['ID'])].dropna()
            new_data.to_sql(table, conn, if_exists='append', index=False)
        print('file updated')
    except Exception as ex: 
        print(f'Table not found, creating temp one\n{ex}')
        df.to_sql(table + ' temp', conn, if_exists='replace', index= False)
        conn.commit()
        conn.close()
    

def export_to_csv(spisok):
    df = pd.DataFrame(spisok)
    df = df.transpose()
    df.to_csv('Комменты.csv', index=False)


def add_update():
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    query_update_columns = """
        UPDATE Comments
        SET City = (SELECT City FROM Users WHERE Users.ID = Comments.Пользователь),
            Country = (SELECT Country FROM Users WHERE Users.ID = Comments.Пользователь),
            Sex = (SELECT Sex FROM Users WHERE Users.ID = Comments.Пользователь);
    """
    cursor.execute(query_update_columns)
    conn.commit()
    # SQL query to retrieve data from the first table (comments)
    comments_query = "SELECT * FROM Comments"
    # Read data into a DataFrame
    comments_df = pd.read_sql_query(comments_query, conn)
    # Group by post ID and calculate the overall sentiment for each post
    posts_df = comments_df.groupby('Пост')['Sentiment'].agg(lambda x: x.mode()[0]).reset_index()
    # Update the second table (posts) with the calculated post sentiment
    for index, row in posts_df.iterrows():
        post_id = row['Пост']
        post_sentiment = row['Sentiment']
        update_query = "UPDATE Posts SET Sentiment = ? WHERE ID = ?"
        conn.execute(update_query, (post_sentiment, post_id))
    # Commit the changes
    conn.commit()
    # Close the connection
    conn.close()

posts, posts_with_comments = get_group_posts(ostin_id, 100)
comments, users = get_comments(ostin_id, posts_with_comments)
export_to_db(posts, 'Posts')
export_to_db(comments, 'Comments')
export_to_db(users, 'Users')
add_update()
