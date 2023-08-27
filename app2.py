from portfolio import *
aux_df = pd.read_csv('stocks.csv', encoding='latin-1')

key = 0

@st.cache_resource
def random_portfolio(df, N):
    return [df[df.columns[0]][np.random.randint(low=0, high=len(df))] for i in range(N)]



with st.sidebar:
    st.title('Portfolio Dashboard')
    N = st.number_input('Number of stocks', min_value=2, step=1)
    stockList = []
    aux_placeholders = aux_df[aux_df.columns[0]][:int(N)]
    with st.expander('Stocks'):
        for i in range(int(N)):
            stockList.append(st.text_input(f'Ticker {i+1}', placeholder=aux_placeholders.iloc[i], key=f'i{key}'))
            key+=1

    stockList = [i.upper() for i in stockList]

    survey = ss.StreamlitSurvey()
    Q1 = ss.Radio(survey, 'Black-Litterman', options=['No', 'Yes'], horizontal=True)
    Q1_value = Q1.display()
    views = {}
    if Q1_value == 'Yes':
        with st.expander('Black-Litterman Views'):
            views_nbr = st.slider('How much views you will add?', min_value=1, max_value=5, value=1)
            for i in range(views_nbr):
                views[i]={}
                views[i]['Cr'] = st.text_input(f'Ticker which will go up', placeholder=aux_placeholders.iloc[i], key=f'i{key}')
                key+=1
                views[i]['Dr'] = st.text_input(f'Ticker which will go down', placeholder=aux_placeholders.iloc[i+1], key=f'i{key}')
                key+=1
                views[i]['i'] = st.number_input(f'Rate', min_value=0.00, value=0.05, key=f'i{key}')
                key+=1

    today = datetime.today()
    end_date = today - timedelta(days=1)
    start_date = today - timedelta(days=365)

    start = st.date_input('Start date', value=start_date, max_value=(today - timedelta(days=9)))
    end = st.date_input('End date', value=end_date, max_value=end_date)
    if end_date>=today:
        st.error('Error: End date must be to yesterday.') 

    Capital = st.number_input('Capital', min_value=1000, step=100)
    _Capital = Capital
    DELTA = st.number_input('Re-allocation time', min_value=1, max_value=(end-start).days,step=1)

    ts = st.number_input('Days to forecast', min_value=DELTA, step=10)

    option = st.selectbox(
                'Which benchmark you will choose?',
                ('S&P 500', 'Nasdaq', 'Dow Jones', 'Invesco')
            )

    optimize = st.form(key='optimize')
    optimize_button = optimize.form_submit_button(label='Optimize')


st.title('Portfolio Composition')
if optimize_button:
    with st.spinner(text='We are optimizing your portfolio...'):
        (Capital, allocation, portfolio, ss) = asset_allocation(stockList, start, end, views, Capital=Capital, DELTA=int(DELTA))

        f = yf.download(stockList, start=start, end=end, interval='1d')['Adj Close']
        f['Portfolio'] = f.mean(axis=1)
        benchmark = yf.download(composite_dict(option), start=start, end=end, interval='1d')['Adj Close']

        (price_paths, mean_end_price) = gbm_stock(f.Portfolio, timesteps=ts, paths=1000)

    rend = round(((Capital/_Capital)-1)*100,2)

    st.metric(label=f'{(end-start).days} days return', value='Overview', delta=str(rend)+'%')
    st.line_chart(f.Portfolio)

    st.subheader('Final Capital')
    st.text('$'+str(round(Capital,2)))

    st.subheader('Final Allocation')
    st.table(allocation)

    st.subheader('Final Distribution')
    st.table(portfolio)

    with st.expander('Forecasting'):                

        rend = round(((mean_end_price/f.Portfolio[-1])-1)*100,2)

        st.text('Last Price: $'+str(round(f.Portfolio[-1],2)))
        st.metric(label=f'${round(mean_end_price,2)} Mean Portfolio Price', value=f'Price for the next {DELTA} days', delta=str(rend)+'%')
        #st.altair_chart(price_paths, use_container_width=True)
        st.line_chart(price_paths)

    with st.expander('Analysis'): 
        st.title('')

        port = pd.DataFrame(
            index=ss['Period'], 
            columns=stockList, 
            data=ss['allocated']
        )
        port= port.bfill()
        port.index = list(map(lambda x: str(x).split(' ')[0], port.index))
        port = port.reset_index().rename(columns={'index': 'date'})
        port.date = pd.to_datetime(port.date, yearfirst=True)

        f.index = list(map(lambda x: str(x).split(' ')[0], f.index))
        returns = f.pct_change(axis=0)
        f = f.reset_index().rename(columns={'index': 'date'})
        f.date = pd.to_datetime(f.date, yearfirst=True)

        st.subheader('Drawdown')
        st.area_chart(drawdown(returns.Portfolio))

        st.subheader('Max Drawdown')
        st.area_chart((returns.Portfolio).rolling(DELTA).apply(max_drawdown))

        st.subheader('Information Ratio')

        
        benchmark.index = list(map(lambda x: str(x).split(' ')[0], benchmark.index))
        benchmark_returns = benchmark.pct_change()
        benchmark_returns.index = pd.to_datetime(benchmark_returns.index, yearfirst=True)

        f = f.drop('Portfolio', axis=1)

        df = pd.merge_asof(
            f, 
            port, 
            on='date',
            direction='nearest'
        )

        f = pd.DataFrame(
            index=f.date,
            columns=stockList,
            data=df.iloc[:,1:len(stockList)+1].values * df.iloc[:,len(stockList)+1:].values
        )

        # compute the portfolio value over time
        portfolio_value = f.sum(axis=1)

        # compute the portfolio daily pnl
        portfolio_pnl = f.sum(axis=1) - f.sum(axis=1).shift()

        # compute the portfolio daily return
        portfolio_returns = (portfolio_pnl / portfolio_value)
        portfolio_returns.name = 'Portfolio'
        benchmark_returns.name = option

        # create cumulative returns
        portfolio_cumulative_returns = (portfolio_returns.fillna(0.0) + 1).cumprod()
        benchmark_cumulative_returns = (benchmark_returns.fillna(0.0) + 1).cumprod()

        ir = round(information_ratio(portfolio_returns, benchmark_returns)*100,2)

        st.metric(
            label=f'{(end-start).days} Days Return',
            value=f'Portfolio vs {option}',
            delta=str(ir)+'%'
            )
        
        st.line_chart(pd.concat([portfolio_cumulative_returns, benchmark_cumulative_returns], axis=1))

else:
    st.write('Choose your portfolios structure')