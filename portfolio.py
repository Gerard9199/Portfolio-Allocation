import pandas as pd
import numpy as np
import yfinance as yf
import cvxpy as cv
import streamlit as st
import streamlit_survey as ss
from datetime import datetime, timedelta

class Portfolio:
    def __init__(self, TICKETS, RF, START, END, MODEL='Mean Variance', VIEWS=None):
        self.TICKETS = TICKETS
        self.RF = RF
        self.START = START
        self.END = END
        self.MODEL = MODEL
        
    def composition_output(self, risk, Portafolio):
        self.risk = risk
        self.Portafolio = Portafolio
        
        print(f'El Sharpe Ratio es de: {round(self.sharpe_ratio(risk), 4)}')
        print(f'El rendimiento anual del portafolio es de: {round(self.risk[0][0], 4)}')
        print(f'El riesgo del portafolio es de: {round(self.risk[1][0], 4)}')
        print('La composición del portafolio es:')
        print(self.Portafolio)
        
    def sharpe_ratio(self, risk):
        self.risk = risk
        return risk[0][0]/risk[1][0]
        
    def p_q_construction(self, VIEWS):
        self.VIEWS = VIEWS
        P = np.zeros(shape=(len(VIEWS), len(self.TICKETS)), dtype=np.int)
        Q = np.zeros(shape=(len(VIEWS),1), dtype=np.float)

        for i in range(1, len(VIEWS)+1):
            #Cr
            try:
                stock_cr = VIEWS[i]['Cr']
                P[i-1][self.TICKETS.index(stock_cr)] = 1
            except:
                pass
            #Dr
            try:
                stock_dr = VIEWS[i]['Dr']
                P[i-1][self.TICKETS.index(stock_dr)] = -1
            except:
                pass
            #i
            stock_i = VIEWS[i]['i']
            Q[i-1] = stock_i

        return (P,Q)
    
    def extract_data(self):
        return yf.download(self.TICKETS, start=self.START, end=self.END, interval='1d')['Adj Close']
    
    def stock_returns(self, data):
        self.data = data
        return data.pct_change().dropna()
        
    def sharpe_ratio_mean_variance(self, returns, BL=None, mu=None, sigma=None):
        self.returns = returns
        self.BL = BL
        self.mu = mu
        self.sigma = sigma
        try:
            if mu == None:
                mu=np.array(np.mean(returns,axis=0), ndmin=2)
                sigma=np.cov(returns.T)
        except:
            pass
        
        w = cv.Variable(mu.shape[1])
        k = cv.Variable(1)
        rf = cv.Parameter(nonneg=True)
        rf.value = self.RF
        u = np.ones((1,mu.shape[1]))*rf

        prob = cv.Problem(cv.Minimize(cv.quad_form(w,sigma)),
                       [(mu-u) @ w == 1,
                       w >= 0,
                       k >= 0,
                       w >= 0.01*k, #para que el peso minimo sea 0.01%
                       cv.sum(w) == k])
        prob.solve(solver=cv.ECOS)
        w_MV = np.array(w.value/k.value, ndmin=2)
        
        Portafolio = pd.DataFrame(data=w_MV,columns=self.TICKETS)
        SR_MV = []
        SR_MV.append([(mu @ w_MV.T * 252).item()])
        SR_MV.append([np.sqrt(w_MV @ sigma @ w_MV.T * 252).item()])
        
        if self.BL == None:
            return (w_MV, SR_MV, Portafolio)
        else:
            return (mu, sigma, w_MV, SR_MV, Portafolio)
    
    def MAD(self, returns):
        self.returns = returns
        return np.mean(np.absolute(returns - np.mean(returns,axis=0)), axis=0).item()
    
    def sharpe_ratio_mean_absolute_desviation(self, returns):
        self.returns = returns
        
        mu = np.array(np.mean(returns,axis=0), ndmin=2)
        w = cv.Variable(returns.shape[1])
        Y = cv.Variable(returns.shape[0])
        k = cv.Variable(1)
        rf = cv.Parameter(nonneg=True)
        rf.value = self.RF
        T = cv.Parameter(nonneg=True)
        T.value = returns.shape[0]
        risk = sum(Y)/T
        u = np.ones((len(returns.values),1))*mu
        a = returns.values - u

        prob = cv.Problem(cv.Minimize(risk),
                                    [a @ w >= -Y,
                                     a @ w <= Y,
                                     Y >= 0,
                                     cv.sum(w) == k,
                                     mu @ w - rf * k == 1,
                                     w >= 0.01*k, #restriccion adicional para pesos mayores a 1%
                                     w >= 0])

        prob.solve(solver=cv.ECOS)
        w_MAD = np.array(w.value/k.value, ndmin=2)

        Portafolio = pd.DataFrame(data=w_MAD,columns=self.TICKETS)
        SR_MAD = []
        SR_MAD.append([(mu @ w_MAD.T * 252).item()])
        SR_MAD.append([ self.MAD(returns.values @ w_MAD.T) * 252 ])
        
        return (w_MAD, SR_MAD, Portafolio)
        
    def CVaR_Hist(self, returns, alpha):
        self.returns = returns
        self.alpha = alpha
        
        sorted_returns = np.sort(returns, axis=0)
        index = int((1-alpha) * len(sorted_returns))
        sum_var = sorted_returns[0]
        for i in range(1, index):
            sum_var += sorted_returns[i]
        return np.abs(sum_var / index).item()
    
    def mean_cvar_hist(self, returns):
        self.returns = returns
        
        mu = np.array(np.mean(returns,axis=0), ndmin=2)
        w = cv.Variable(returns.shape[1])
        n = cv.Parameter(nonneg=True)
        n.value = returns.shape[0]
        VaR = cv.Variable(1)
        alpha = cv.Parameter(nonneg=True)
        alpha.value=0.99
        Z = cv.Variable(returns.shape[0])
        risk = VaR+1/((1-alpha)*n)*cv.sum(Z)
        k = cv.Variable(1)
        rf = cv.Parameter(nonneg=True)
        rf.value = self.RF
        X = returns.values @ w

        prob = cv.Problem(cv.Minimize(risk),
                            [Z >= 0,
                             Z >= -X - VaR,
                             mu @ w - rf * k == 1,
                             cv.sum(w) == k,
                             w >= 0.01*k, #pesos minimos de 1%
                             w >= 0,
                             k>=0])

        prob.solve(solver=cv.ECOS)
        w_CVaR_Hist=np.array(w.value/k.value, ndmin=2)

        Portafolio = pd.DataFrame(data=w_CVaR_Hist,columns=self.TICKETS)
        SR_CVaR_Hist=[]
        SR_CVaR_Hist.append([(mu @ w_CVaR_Hist.T*252).item()])
        SR_CVaR_Hist.append([(self.CVaR_Hist(returns.values @ w_CVaR_Hist.T,0.99)*np.sqrt(252))])
        
        return (w_CVaR_Hist, SR_CVaR_Hist, Portafolio)
        
    def MC_Corr_Sample(self, returns, N):
        self.returns = returns
        self.N = N
        
        mu=np.array(np.mean(returns,axis=0), ndmin=2)
        cols = returns.shape[1]               
        np.random.seed(0)
        observations = np.random.normal(0, 1, (cols, N)) 
        cov_matrix = np.cov(returns.T)   

        Chol = np.linalg.cholesky(cov_matrix) # Descomposicion de Cholesky 
        sam_eq_mean = Chol.dot(observations)             
        samples = sam_eq_mean.transpose() + mu 
        return samples

    def MC_Sample(self, returns, N):
        self.returns = returns
        self.N = N
        
        mu=np.array(np.mean(returns), ndmin=2)
        sd=np.array(np.std(returns), ndmin=2)
        np.random.seed(0)
        observations = np.random.randn(N, 1)
        sam_eq_mean = observations*sd            
        samples = sam_eq_mean + mu      
        return samples    
        
    def CVaR_MC(self, returns, alpha, N):
        self.returns = returns
        self.alpha = alpha
        self.N = N
        
        returns=self.MC_Sample(returns, N)
        sorted_returns = np.sort(returns, axis=0)
        index = int((1-alpha) * len(sorted_returns))
        sum_var = sorted_returns[0]
        for i in range(1, index):
            sum_var += sorted_returns[i]
        return np.abs(sum_var / index).item()    
        
    def mean_cvar_montecarlo(self, returns):
        self.returns = returns
        
        mu = np.array(np.mean(returns,axis=0), ndmin=2)
        returns_MC=self.MC_Corr_Sample(returns, 1000)
        w = cv.Variable(returns_MC.shape[1])
        n = cv.Parameter(nonneg=True)
        n.value = returns_MC.shape[0]
        VaR = cv.Variable(1)
        alpha = cv.Parameter(nonneg=True)
        alpha.value=0.99
        Z = cv.Variable(returns_MC.shape[0])
        risk = VaR+1/((1-alpha)*n)*cv.sum(Z)
        k = cv.Variable(1)
        rf = cv.Parameter(nonneg=True)
        rf.value = self.RF
        X = returns_MC @ w

        #definiendo el problema, funcion objetivo y reestricciones
        prob = cv.Problem(cv.Minimize(risk),
                            [Z >= 0,
                             Z >= -X-VaR,
                             mu @ w - rf * k == 1,
                             cv.sum(w) == k,
                             w >= 0.01*k, #pesos minimos de 1%
                             w >= 0,
                             k>=0])

        #resolviendo el problema
        prob.solve(solver=cv.ECOS)
        w_CVaR_MC = np.array(w.value/k.value, ndmin=2)
        Portafolio = pd.DataFrame(data=w_CVaR_MC,columns=self.TICKETS)
        #almacenando la data
        SR_CVaR_MC=[]
        SR_CVaR_MC.append([(mu @ w_CVaR_MC.T * 252).item()])
        SR_CVaR_MC.append([self.CVaR_MC(returns.values @ w_CVaR_MC.T,0.99,1000)*np.sqrt(252)])
        
        return (w_CVaR_MC, SR_CVaR_MC, Portafolio)
    
    def black_litterman(self, returns, P, Q, tau, delta, historical=None):
        self.returns = returns
        self.P = P
        self.Q = Q
        self.tau = tau
        self.delta = delta
        self.historical = historical
        
        riskfree = 0
        (mu, sigma, weight, risk, Portfolio) = self.sharpe_ratio_mean_variance(returns, BL=True)
        
        w = np.matrix(weight)
        P = np.matrix(P)
        Q = np.matrix(Q)
        S = np.matrix(sigma)
        Omega = np.matrix(np.diag(np.diag(P * (tau * S) * P.T)))

        if historical == 'Historical':
            PI = mu.T - riskfree
        else:
            PI = delta * S * w.T

        M = ((tau * S).I + P.T * Omega.I * P).I
        PI_1 = ((tau * S).I + P.T * Omega.I * P).I * ((tau * S).I * PI + P.T * Omega.I * Q)
        S_1 = S + M
        w_1 = (1/(1 + tau)) * (delta * S_1).I * PI_1
        mu = PI_1 + riskfree
        
        (w_MAD, SR_MAD, Portafolio) = self.sharpe_ratio_mean_variance(returns, mu=np.matrix(mu.T), sigma=S_1)
        
        w_BL = w_MAD
        Portafolio = pd.DataFrame(data=w_BL,columns=self.TICKETS)

        SR_BL=[]
        SR_BL.append([(mu.T @ w_BL.T * 252).item()])
        SR_BL.append([np.sqrt(w_BL @ S_1 @ w_BL.T * 252).item()])
        
        return (w_BL, SR_BL, Portafolio)
    
    def kurt_matrix(self, returns):
        self.returns = returns
        
        P = returns.to_numpy()
        T, n = P.shape
        mu = np.mean(P, axis=0).reshape(1,-1)
        mu = np.repeat(mu, T, axis=0)
        x = P - mu
        ones = np.ones((1,n))
        z = np.kron(ones, x) * np.kron(x, ones);
        S4 = 1/T * z.T @ z
        return S4

    def block_vec_pq(self, A, p, q):
        self.A = A
        self.p = p
        self.q = q
        
        mp, nq = A.shape
        if mp % p == 0 and nq % q == 0:
            m = int(mp / p)
            n = int(nq / q)
            bvec_A = np.empty((0, p * q))
            for j in range(n):
                Aj = np.empty((0, p * q))
                for i in range(m):
                    Aij = (
                        A[i * p : (i + 1) * p, j * q : (j + 1) * q]
                        .reshape(-1, 1, order="F")
                        .T
                    )
                    Aj = np.vstack([Aj, Aij])
                bvec_A = np.vstack([bvec_A, Aj])
        return bvec_A
    
    def kurtosis_optimization(self, returns):
        self.returns = returns
        
        T, n = returns.shape
        K = 2*n
        mu=np.array(np.mean(returns,axis=0), ndmin=2)
        sigma=np.cov(returns.T)
        Sigma_4 = self.kurt_matrix(returns)

        A = self.block_vec_pq(Sigma_4, n, n)
        s_A, V_A = np.linalg.eig(A)
        s_A = np.clip(s_A, 0, np.inf)
        Bi = []
        for i in range(K):
            B = s_A[i] ** 0.5 * V_A[:, i]
            B = B.reshape((n, n), order="F").real
            Bi.append(B)

        rows = int(n*(n+1)/2)
        w = cv.Variable((n,1))
        z = cv.Variable((rows,1))
        X = cv.Variable((n,n), PSD=True)
        g = cv.Variable((K, 1))
        risk = cv.pnorm(g,p=2)
        M = cv.bmat([[X, w], [w.T, np.ones((1, 1))]])
        constraints = [cv.sum(w) == 1,
                       w >= 0.01,
                       M >> 0]

        for i in range(K):
            constraints += [g[i, 0] == cv.trace(Bi[i] @ X)]

        obj = cv.Minimize(risk * 1000)
        prob = cv.Problem(obj, constraints)

        prob.solve()
        w_KT = w.value.T
        Portafolio = pd.DataFrame(data=w_KT,columns=self.TICKETS)
        SR_KT = []
        SR_KT.append([(mu @ w_KT.T * 252).item()])
        SR_KT.append([np.sqrt(w_KT @ sigma @ w_KT.T * 252).item()])
        
        return (w_KT, SR_KT, Portafolio)
    
    def model_execution(self, METHOD=None, VIEWS=None):
        self.METHOD = METHOD
        self.VIEWS = VIEWS
        data = self.extract_data()
        returns = self.stock_returns(data)
        
        if self.MODEL == 'Mean Variance':
            (weight, risk, Portfolio) = self.sharpe_ratio_mean_variance(returns)
            sharpe = round(self.sharpe_ratio(risk), 4)
            return (Portfolio, sharpe)
            #return self.composition_output(risk, Portfolio)
            
        elif self.MODEL == 'Mean Absolute Desviation':
            (weight, risk, Portfolio) = self.sharpe_ratio_mean_absolute_desviation(returns)
            sharpe = round(self.sharpe_ratio(risk), 4)
            return (Portfolio, sharpe)
            #return self.composition_output(risk, Portfolio)       
        
        elif self.MODEL == 'Mean CVaR':
            if self.METHOD == 'Historical': 
                (weight, risk, Portfolio) = self.mean_cvar_hist(returns)
                sharpe = round(self.sharpe_ratio(risk), 4)
                return (Portfolio, sharpe)
                #return self.composition_output(risk, Portfolio)
            
            elif self.METHOD == 'Monte Carlo': 
                (weight, risk, Portfolio) = self.mean_cvar_montecarlo(returns)
                sharpe = round(self.sharpe_ratio(risk), 4)
                return (Portfolio, sharpe)
                #return self.composition_output(risk, Portfolio)
            
        elif self.MODEL == 'Black-Litterman':
            self.VIEWS = VIEWS
            delta = len(VIEWS) #delta, lambda o factor de aversion al riesgo
            tau = 1/returns.shape[0]
            (P,Q) = self.p_q_construction(VIEWS)
            
            if self.METHOD == 'Historical':
                (weight, risk, Portfolio) = self.black_litterman(returns, P, Q, tau, delta, historical=self.METHOD)
            else:
                (weight, risk, Portfolio) = self.black_litterman(returns, P, Q, tau, delta)
                
            sharpe = round(self.sharpe_ratio(risk), 4)
            return (Portfolio, sharpe)
            #return self.composition_output(risk, Portfolio)
        
        elif self.MODEL == 'Kurtosis':
            
            (weight, risk, Portfolio) = self.kurtosis_optimization(returns)
            sharpe = round(self.sharpe_ratio(risk), 4)
            return (Portfolio, sharpe)
            #return self.composition_output(risk, Portfolio)
            
"""
Others
"""

def date_structure(x):
    x = str(x).split('-')
    day = int(x[2])
    month = int(x[1])
    year = int(x[0])
    
    return (year, month, day)

def asset_allocation(stockList, start, end, views, Capital=10000, DELTA=21):
    portfolio = model_selection(stockList, start, end, views)
    prices_t_0 = pd.DataFrame()
    
    (year, month, day) = date_structure(start)
    (_year, _month, _day) = date_structure(end)
    
    _start = datetime(year, month, day)
    _end = datetime(year, month, day) + timedelta(DELTA)
    k = datetime(year, month, day) + timedelta(DELTA+1)
    
    ss = {
    'Period':[],
    'allocated':[],
    }
    while _end <= datetime(_year, _month, _day):
        print(_start)
        print(_end)
        prices_t_1 = yf.download(stockList, start=_end, end=k, interval='1d')['Adj Close'][portfolio.columns].reset_index(drop=True)
        if prices_t_0.empty == False:
            if (allocation*prices_t_1).sum().sum() > (allocation*prices_t_0).sum().sum():
                print('SELL')
                Capital = (allocation*prices_t_1).sum().sum()
                portfolio = model_selection(stockList, _start, _end, views)
                print('BUY')
                allocation = portfolio*Capital / prices_t_1
                print('Allocated')
                ss['Period'].append(str(_end))
                ss['allocated'].append(list(allocation.values[0]))
            else:
                print('HOLD')
                ss['Period'].append(str(_end))
                ss['allocated'].append(list(allocation.values[0]))
                
        else:
            allocation = portfolio*Capital / prices_t_1
            print('Allocated')
            ss['Period'].append(str(_end))
            print(allocation)
            ss['allocated'].append(list(allocation.values[0]))
        
        prices_t_0 = prices_t_1
        _start = _end
        _end = datetime(_start.year, _start.month, _start.day) + timedelta(DELTA)
        k = datetime(_start.year, _start.month, _start.day) + timedelta(DELTA+1)
        
    allocation = allocation.rename(index={0: 'Stocks'})
    portfolio = portfolio.rename(index={0: 'Weights'})
    return (Capital, allocation, portfolio, ss)

def model_selection(stockList, start, end, views):
    BL_H = pd.DataFrame()
    s_BL_H = 0
    
    BL = pd.DataFrame()
    s_BL = 0
    
    M_CVaR_H = pd.DataFrame()
    s_M_CVaR_H = 0
    
    M_CVaR_MC = pd.DataFrame()
    s_M_CVaR_MC = 0
    
    MV = pd.DataFrame()
    s_MV = 0
    
    MAD = pd.DataFrame()
    s_MAD = 0
    
    K = pd.DataFrame()
    s_K = 0
    
    try:
        (BL_H, s_BL_H) = Portfolio(stockList, 0, start, end, 'Black-Litterman').model_execution(METHOD='Historical', VIEWS=views)
    except:
        pass
    
    try:
        (BL, s_BL) = Portfolio(stockList, 0, start, end, 'Black-Litterman').model_execution(VIEWS=views)
    except:
        pass
    
    try:
        (M_CVaR_H, s_M_CVaR_H) = Portfolio(stockList, 0, start, end, 'Mean CVaR').model_execution(METHOD='Historical')
    except:
        pass
    
    try:
        (M_CVaR_MC, s_M_CVaR_MC) = Portfolio(stockList, 0, start, end, 'Mean CVaR').model_execution(METHOD='Monte Carlo')
    except:
        pass
        
    try:
        (MV, s_MV) = Portfolio(stockList, 0, start, end, 'Mean Variance').model_execution()
    except:
        pass
    
    try:
        (MAD, s_MAD) = Portfolio(stockList, 0, start, end, 'Mean Absolute Desviation').model_execution()
    except:
        pass
    
    try:
        (K, s_K) = Portfolio(stockList, 0, start, end, 'Kurtosis').model_execution()
    except:
        pass
    
    sharpe = {
        'Black-Litterman Historical':s_BL_H,
        'Black-Litterman':s_BL,
        'Mean CVaR Historial':s_M_CVaR_H,
        'Mean CVaR Monte Carlo':s_M_CVaR_MC,
        'Mean Variance':s_MV,
        'Mean Absolute Desviation':s_MAD,
        'Kurtosis':s_K,
    }
    
    portfolios = {
        'Black-Litterman Historical':BL_H,
        'Black-Litterman':BL,
        'Mean CVaR Historial':M_CVaR_H,
        'Mean CVaR Monte Carlo':M_CVaR_MC,
        'Mean Variance':MV,
        'Mean Absolute Desviation':MAD,
        'Kurtosis':K,
    }
    
    model = list(sharpe.keys())[list(sharpe.values()).index(max(sharpe.values()))]
    print(model)
    return portfolios[model]

def wiener_process(delta, sigma, time, paths):
    # return an array of samples from a normal distribution
    return sigma * np.random.normal(loc=0, scale=np.sqrt(delta), size=(time, paths))

def gbm_returns(delta, sigma, time, mu, paths):
    process = wiener_process(delta, sigma, time, paths)
    return np.exp(
        process + (mu - sigma**2 / 2) * delta
    )

def gbm_levels(s0, delta, sigma, time, mu, paths):
    returns = gbm_returns(delta, sigma, time, mu, paths)
    stacked = np.vstack([np.ones(paths), returns])
    return s0 * stacked.cumprod(axis=0)

def gbm_stock(portfolio, timesteps, paths=10000):
    stock_log_price = np.log(portfolio)
    stock_log_return = stock_log_price.diff()
    s0 = portfolio[-1]
    sigma = np.std(stock_log_price)
    mu = stock_log_return.sum()/stock_log_return.count()
    mu = mu*252 + 0.5*stock_log_return.var()*np.sqrt(252)
    delta = 1.0/252.0
    
    price_paths = gbm_levels(s0, delta, sigma, timesteps, mu, paths)
    mean_end_price = np.mean(price_paths)
    
    return (price_paths, mean_end_price)

def drawdown(returns):
    returns = returns.fillna(0.0)
    cumulative = (returns + 1).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    return (cumulative - running_max) / running_max

def max_drawdown(returns):
    return np.min(drawdown(returns))

def information_ratio(portfolio_returns, benchmark_returns):
    active_return = portfolio_returns - benchmark_returns
    tracking_error = active_return.std()
    return active_return.mean() / tracking_error

def composite_dict(x):
    composite = {
        'Invesco':'QQQ', 
        'S&P 500':'^GSPC', 
        'Nasdaq':'^IXIC', 
        'Dow Jones':'^DJI',
    }

    return composite[x]

