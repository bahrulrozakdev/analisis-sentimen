import pandas as pd
from textblob import TextBlob
import plotly.express as px
from wordcloud import WordCloud
import streamlit as st
import matplotlib.pyplot as plt
import io

data = pd.read_csv('data_pengisian_form.csv')

comment_columns = [col for col in data.columns if 'Berikan alasan terbaik' in col]
platform_columns = [col for col in data.columns if 'Sejauh ini kakak sering bertanya di platform' in col]

st.title('Analisis Sentimen dan Visualisasi Data')

selected_comment_col = st.selectbox('Pilih Kolom Komentar:', comment_columns)
alasan_col = selected_comment_col

selected_platform_col = st.selectbox('Pilih Kolom Platform:', platform_columns)
platform_col = selected_platform_col

unique_platforms = data[platform_col].dropna().unique()
selected_platforms = st.multiselect('Pilih Platform untuk Filter:', unique_platforms, default=unique_platforms)

filtered_data = data[data[platform_col].isin(selected_platforms)] if selected_platforms else data

def analyze_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    sentiment = 'Positif' if polarity > 0 else 'Negatif' if polarity < 0 else 'Netral'
    return polarity, sentiment

filtered_data[['Sentiment_Polarity', 'Sentiment_Category']] = filtered_data[alasan_col].apply(lambda x: pd.Series(analyze_sentiment(x)))

st.subheader('Data dengan Analisis Sentimen')
st.dataframe(filtered_data[['Sentiment_Polarity', 'Sentiment_Category', alasan_col]].head())

average_polarity = filtered_data['Sentiment_Polarity'].mean()
st.write(f"Rata-Rata Polaritas Sentimen: {average_polarity:.2f}")

st.subheader('Distribusi Kategori Sentimen Komentar')
sentiment_counts = filtered_data['Sentiment_Category'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment_Category', 'Jumlah_Komentar']
fig_sentiment_category = px.bar(sentiment_counts,
                               x='Sentiment_Category', y='Jumlah_Komentar',
                               labels={'Sentiment_Category': 'Kategori Sentimen', 'Jumlah_Komentar': 'Jumlah Komentar'},
                               title='Distribusi Kategori Sentimen Komentar',
                               color='Sentiment_Category',
                               color_discrete_map={'Positif': 'green', 'Negatif': 'red', 'Netral': 'gray'})
fig_sentiment_category.update_layout(xaxis_title='Kategori Sentimen', yaxis_title='Jumlah Komentar')
st.plotly_chart(fig_sentiment_category)

st.subheader('Distribusi Polaritas Sentimen Komentar')
fig_sentiment_polarity = px.histogram(filtered_data, x='Sentiment_Polarity',
                                      labels={'Sentiment_Polarity': 'Skor Polaritas Sentimen'},
                                      title='Distribusi Polaritas Sentimen Komentar',
                                      nbins=20,
                                      color_discrete_sequence=['purple'])
fig_sentiment_polarity.update_layout(xaxis_title='Skor Polaritas Sentimen', yaxis_title='Frekuensi')
st.plotly_chart(fig_sentiment_polarity)

st.subheader('Frekuensi Penggunaan Platform')
platform_counts = filtered_data[platform_col].value_counts().reset_index()
platform_counts.columns = [platform_col, 'Jumlah_Responden']
fig_platform_usage = px.bar(platform_counts,
                           x=platform_col, y='Jumlah_Responden',
                           labels={platform_col: 'Platform', 'Jumlah_Responden': 'Jumlah Responden'},
                           title='Frekuensi Penggunaan Platform',
                           color=platform_col,
                           color_discrete_sequence=px.colors.qualitative.Plotly)
fig_platform_usage.update_layout(xaxis_title='Platform', yaxis_title='Jumlah Responden')
st.plotly_chart(fig_platform_usage)

text = ' '.join(filtered_data[alasan_col].dropna())
max_words = st.slider('Jumlah Kata Maksimum pada Word Cloud:', min_value=50, max_value=500, value=200, step=10)
wordcloud = WordCloud(width=800, height=400,
                      background_color='white',
                      colormap='viridis',
                      max_words=max_words).generate(text)

st.subheader('Word Cloud dari Komentar')
fig_wordcloud = plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud dari Komentar')
st.pyplot(fig_wordcloud)

st.subheader('Analisis Sentimen Teks Tambahan')
user_input = st.text_area("Masukkan Teks untuk Analisis Sentimen:")
if user_input:
    polarity, sentiment = analyze_sentiment(user_input)
    st.write(f"Polaritas Sentimen: {polarity:.2f}")
    st.write(f"Kategori Sentimen: {sentiment}")

st.subheader('Unduh Data')
csv = filtered_data.to_csv(index=False)
st.download_button(label='Unduh Data yang Dianalisis sebagai CSV',
                   data=csv,
                   file_name='data_analisis.csv',
                   mime='text/csv')
