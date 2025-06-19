import mysql.connector

def __get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='emotion_recognition'
    )
    return connection
