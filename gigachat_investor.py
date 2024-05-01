from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat
import yfinance as yf
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import ast
import json

chat = GigaChat(model="GigaChat-Pro",
                credentials="ТОКЕН",
                verify_ssl_certs=False)

def get_article_text(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_text = ' '.join([p.get_text() for p in soup.find_all('p')])
        return article_text
    except:
        return "Error retrieving article text."


def get_stock_data(ticker, years):
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=years * 365)

    stock = yf.Ticker(ticker)

    hist_data = stock.history(start=start_date, end=end_date)

    balance_sheet = stock.balance_sheet

    financials = stock.financials

    news = stock.news

    return hist_data, balance_sheet, financials, news


def get_sentiment_analysis(ticker, news):
    system_prompt = f"Вы являетесь помошником по анализу настроений. Проанализируйте настроения в новостных статьях для {ticker} и предоставьте краткую информацию об общих настроениях и любых заметных изменениях с течением времени. Будьте взвешенным и проницательным. Вы скептически настроенный инвестор."

    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]

    news_text = ""
    for article in news:
        article_text = get_article_text(article['link'])
        timestamp = datetime.fromtimestamp(article['providerPublishTime']).strftime("%Y-%m-%d")
        news_text += f"\n\n---\n\nDate: {timestamp}\nTitle: {article['title']}\nText: {article_text}"

    mes = f"Новостные статьи для {ticker}:\n{news_text}\n\n----\n\nСодержат краткую информацию об общем настроении и любых заметных изменениях с течением времени"

    messages.append(HumanMessage(content=mes))
    res = chat(messages)
    print(res.content)
    return res.content


def get_analyst_ratings(ticker):
    stock = yf.Ticker(ticker)
    recommendations = stock.recommendations
    if recommendations is None or recommendations.empty:
        return "No analyst ratings available."
    latest_rating = recommendations.iloc[-1]
    firm = stock.info.get("longName", "'N/A")
    info = latest_rating
    action = determine_action(latest_rating.get("strongBuy"), latest_rating.get("buy"), latest_rating.get("hold"),
                              latest_rating.get("sell"), latest_rating.get("strongSell"))

    rating_summary = f"Анализ для {ticker}:\nFirm: {firm}\nИнфо: {info}\nТенденция: {action}"

    return rating_summary


def get_industry_analysis(ticker):
    stock = yf.Ticker(ticker)
    industry = stock.info['industry']
    sector = stock.info['sector']

    system_prompt = f"Вы являетесь помощником по отраслевому анализу. Проведите анализ отрасли {industry} и сектора {sector}, включая тенденции, перспективы роста, изменения в законодательстве и конкурентную среду. Будьте взвешенными и проницательными. Подумайте о положительных и отрицательных сторонах акций. Будьте уверены в своем анализе. Вы скептически настроенный инвестор."
    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]
    mes = f"Представьте анализ отрасли {industry} и сектора {sector}"
    messages.append(HumanMessage(content=mes))
    res = chat(messages)
    print(res.content)
    return res.content


def get_final_analysis(ticker, comparisons, sentiment_analysis, analyst_ratings, industry_analysis):
    system_prompt = f"Вы финансовый аналитик, дающий окончательную инвестиционную рекомендацию для {ticker} на основе предоставленных данных и анализа. Будьте взвешенными и разборчивыми. По-настоящему подумайте о положительных и отрицательных сторонах акций. Будьте уверены в своем анализе. Вы скептически настроенный инвестор."
    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]
    mes = f"Ticker: {ticker}\n\nСравнительный анализ:\n{json.dumps(comparisons, indent=2)}\n\nАнализ настроений:\n{sentiment_analysis}\n\nОценки аналитиков:\n{analyst_ratings}\n\nАнализ отрасли:\n{industry_analysis}\n\nНа основании предоставленных данных и анализов, пожалуйста, предоставьте комплексный анализ инвестиций и рекомендацию для {ticker}. Учитывайте финансовую силу компании, перспективы роста, конкурентное положение и потенциальные риски. Предложите четкое и лаконичное предложение о том, стоит ли покупать, держать или продавать акции, вместе с подтверждающими аргументами."

    messages.append(HumanMessage(content=mes))
    res = chat(messages)
    print(res.content)

    return res.content


def generate_ticker_ideas(industry):
    system_prompt = f"Вы - ассистент финансового аналитика. Создайте список из 5 символов для основных компаний в отрасли {industry} в виде списка, который можно анализировать на Python"
    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]
    mes = f"Пожалуйста, предоставьте список из 5 символов для обозначения основных компаний в отрасли {industry} в виде списка, доступного для анализа на Python. В ответ укажите только список, никакого другого текста"
    messages.append(HumanMessage(content=mes))
    res = chat(messages)

    ticker_list = ast.literal_eval(res.content)
    return [ticker.strip() for ticker in ticker_list]


def get_current_price(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period='1d', interval='1m')
    return data['Close'][-1]


def rank_companies(industry, analyses, prices):
    system_prompt = f"Вы финансовый аналитик, составляющий рейтинг компаний в отрасли на основе их инвестиционного потенциала. Будьте проницательны и сообразительны. По-настоящему подумайте о том, ценны ли акции или нет. Вы скептически настроенный инвестор."
    analysis_text = "\n\n".join(
        f"Ticker: {ticker}\nТекущая цена: {prices.get(ticker, 'N/A')}\nАнализ:\n{analysis}"
        for ticker, analysis in analyses.items()
    )
    print(analysis_text)
    messages = [
        SystemMessage(
            content=system_prompt
        )
    ]

    mes = f"Отрасль: {industry}\n\nАнализы компаний:\n{analysis_text}\n\nНа основе предоставленных анализов, пожалуйста, оцените компании по привлекательности для инвестиций, от наиболее привлекательной до наименее привлекательной. Предоставьте полное обоснование вашей оценки. В каждом обосновании укажите текущую цену (если она доступна) и целевую цену."

    messages.append(HumanMessage(content=mes))
    res = chat(messages)
    print(res.content)

    return res.content


def determine_action(strong_buy, buy, hold, sell, strong_sell):
    """
    Определяет действие с акцией на основе предоставленных данных.

    :param int strong_buy: количество сильных покупок
    :param int buy: количество покупок
    :param int hold: количество удержаний
    :param int sell: количество продаж
    :param int strong_sell: количество сильных продаж
    :return: Строка с рекомендованным действием ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')
    """
    # Суммируем все положительные рекомендации
    positive_recommendations = strong_buy + buy

    # Суммируем все нейтральные рекомендации
    neutral_recommendations = hold

    # Суммируем все отрицательные рекомендации
    negative_recommendations = sell + strong_sell

    # Вычисляем общее количество рекомендаций
    total_recommendations = positive_recommendations + neutral_recommendations + negative_recommendations

    # Если нет никаких рекомендаций, возвращаем 'none'
    if total_recommendations == 0:
        return 'none'

    # Рассчитываем долю положительных рекомендаций
    positive_percentage = round(positive_recommendations / total_recommendations * 100,
                                2) if total_recommendations > 0 else 0

    # Рассчитываем долю нейтральных рекомендаций
    neutral_percentage = round(neutral_recommendations / total_recommendations * 100,
                               2) if total_recommendations > 0 else 0

    # Рассчитываем долю отрицательных рекомендаций
    negative_percentage = round(negative_recommendations / total_recommendations * 100,
                                2) if total_recommendations > 0 else 0

    # Если доля положительных рекомендаций больше 50%, рекомендуем сильную покупку
    if positive_percentage >= 50:
        return 'strong_buy'

    # Если доля положительных рекомендаций больше доли нейтральных, но меньше 50%, рекомендуем покупку
    elif positive_percentage > neutral_percentage and positive_percentage < 50:
        return 'buy'

    # Если доля нейтральных рекомендаций больше, рекомендуем удержание
    elif neutral_percentage >= positive_percentage and neutral_percentage >= negative_percentage:
        return 'hold'

    # Если доля отрицательных рекомендаций больше, рекомендуем продажу
    elif negative_percentage > neutral_percentage:
        return 'sell'

    # Если доля отрицательных рекомендаций больше 50%, рекомендуем сильную продажу
    elif negative_percentage >= 50:
        return 'strong_sell'

    # Если ни одно из условий не выполняется, возвращаем 'none'
    return 'none'


# User input
industry = input("Введите индустрию для финансового анализа: ")
years = 1

tickers = generate_ticker_ideas(industry)
print(f"\nИнвестиционные идеи для {industry} индустрии:")
print(", ".join(tickers))

analyses = {}
prices = {}
for ticker in tickers:
    try:
        print(f"\nАнализ {ticker}...")
        hist_data, balance_sheet, financials, news = get_stock_data(ticker, years)

        main_data = {
            'hist_data': hist_data,
            'balance_sheet': balance_sheet,
            'financials': financials,
            'news': news
        }
        sentiment_analysis = get_sentiment_analysis(ticker, news)
        analyst_ratings = get_analyst_ratings(ticker)
        industry_analysis = get_industry_analysis(ticker)
        final_analysis = get_final_analysis(ticker, {}, sentiment_analysis, analyst_ratings, industry_analysis)
        analyses[ticker] = final_analysis
        prices[ticker] = get_current_price(ticker)
    except:
        pass

# Rank the companies based on their analyses
ranking = rank_companies(industry, analyses, prices)
print(f"\nРанжирование и оценка компаний для выбранной отрасли {industry}:")
print(ranking)
