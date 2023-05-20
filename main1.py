import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import datetime
from tkinter.simpledialog import askinteger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplfinance as mpf
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import twstock
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA 
import pandas as pd 


#window========================

class Window(tk.Tk):
    def __init__(self):
        super().__init__()
    #主視窗:設定===
        
        self.geometry('1400x825') 
        self.configure(background='#ffffff')

        # 創建PhotoImage物件
        # self.img = ImageTk.PhotoImage(Image.open("image.png").resize((450, 825), Image.ANTIALIAS))
        # self.img = ImageTk.PhotoImage(Image.open("image.png"))        

        style1 = ttk.Style()
        #style1.configure('bg1.TFrame', background='#fcfced') 
        style1.configure('bg1.TFrame') #要全部系統色就用這個
        style2 = ttk.Style()
        style2.configure('bg2.TFrame', background='#d5eaed')            
        mainFrame1 = ttk.Frame(self, height=825, width=450, relief=tk.RAISED, borderwidth=1, style='bg1.TFrame')
        mainFrame1.grid(row=0, column=0, sticky="nw")
        mainFrame2 = ttk.Frame(self, height=825, width=800, relief=tk.RAISED, borderwidth=1, style='bg2.TFrame')
        mainFrame2.grid(row=0, column=1, sticky="nsew")

        # 設定背景圖(不要背景圖的話就拿掉)
        # bg_label = tk.Label(mainFrame1, image=self.img)
        # bg_label.place(x=170, y=-240, relwidth=1, relheight=1)

    #八大窗格:設定

        # mainFrame1===========================
        self.BigFrame1 = tk.Frame(mainFrame1,height=100,width=450) 
        self.BigFrame1.grid(row=0, column=3, rowspan=2, columnspan=2, padx=30, pady=30,sticky="nw")

        self.BigFrame2 = tk.Frame(mainFrame1,height=900,width=450) 
        self.BigFrame2.grid(row=2, column=3, rowspan=2, columnspan=2, padx=5, pady=5,sticky="nw")

        ## mainFrame2 =========================
        # 用tk.Canvas()加入Scrollbar 
        canvas = tk.Canvas(mainFrame2, height=825, width=800, bg="#d5eaed")
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(mainFrame2, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

        
        # 將mainFrame2的所有self.BigFrame放入convas
        self.BigFrame3 = ttk.LabelFrame(canvas,height=499,width=800,text=f"趨勢圖") #趨勢圖
        canvas.create_window((0, 0), window=self.BigFrame3, anchor="nw")

        self.BigFrame4 = ttk.LabelFrame(canvas,height=750,width=800,text=f"熱力圖") #熱力圖
        canvas.create_window((0, 501), window=self.BigFrame4, anchor="nw")

        self.BigFrame5 = ttk.LabelFrame(canvas,height=600,width=800,text=f"K線圖") #K線圖
        canvas.create_window((0,1252), window=self.BigFrame5, anchor="nw")

        self.BigFrame6 = ttk.LabelFrame(canvas,height=650,width=800,text=f"盒鬚圖") #盒鬚圖
        canvas.create_window((0, 1853), window=self.BigFrame6, anchor="nw")

        self.BigFrame7 = ttk.LabelFrame(canvas,height=800,width=800,text=f"散點圖") #散點圖
        canvas.create_window((0, 2504), window=self.BigFrame7, anchor="nw")

        self.BigFrame8 = ttk.LabelFrame(canvas,height=800,width=800,text=f"績效回測") 
        canvas.create_window((0, 3305), window=self.BigFrame8, anchor="nw")

        self.BigFrame9 = ttk.LabelFrame(canvas,height=800,width=800,text=f"最佳資產策略") 
        canvas.create_window((0, 4105), window=self.BigFrame9, anchor="nw")

        self.BigFrame10 = ttk.LabelFrame(canvas,height=800,width=800,text=f"最佳SQN策略") 
        canvas.create_window((0, 4905), window=self.BigFrame10, anchor="nw")

        # 在BigFrame3和BigFrame4之間加入一條水平線
        canvas.create_line(0, 500, 800, 500, fill="#999999", width=1)
        
        # 在BigFrame4和BigFrame5之間加入一條水平線
        canvas.create_line(0, 1251, 800, 1251, fill="#999999", width=1)
        
        # 在BigFrame5和BigFrame6之間加入一條水平線
        canvas.create_line(0, 1852, 800, 1852, fill="#999999", width=1)

        # 在BigFrame6和BigFrame7之間加入一條水平線
        canvas.create_line(0, 2503, 800, 2503, fill="#999999", width=1)

        # 在BigFrame7和BigFrame8之間加入一條水平線
        canvas.create_line(0, 3304, 800, 3304, fill="#999999", width=1)

        # 在BigFrame8和BigFrame9之間加入一條水平線
        canvas.create_line(0, 4104, 800, 4104, fill="#999999", width=1)

        # 在BigFrame9和BigFrame10之間加入一條水平線
        canvas.create_line(0, 4904, 800, 4904, fill="#999999", width=1)

#《第一視窗：輸入》========================


    #輸入視窗================
        inputFrame = tk.Frame(self.BigFrame1)
        tk.Label(inputFrame,text="股票投資有賺有賠，\n看看就好看看就好。",font=("Arial",20,'bold'),fg="#76529c").pack()
        inputFrame.pack()
        

    #代號輸入===============
        self.stockframe = tk.Frame(self.BigFrame1,width=450)
        self.stockframe.pack()
        
        stocklabel=tk.Label(self.stockframe, text="輸入股票號碼   :",font=('Arial',15))
        stocklabel.grid(row=0,column=0,sticky=tk.W)
        
        #加上預設值，測試方便=============
        stockID_default = '2330'
        self.stockIDvar = tk.StringVar(value=stockID_default)
        
        #self.stockIDentry  = tk.Entry(self.stockframe,text=tk.StringVar(),bd=5)#不要預設值的話改回這行
        self.stockIDentry  = tk.Entry(self.stockframe,textvariable=self.stockIDvar,bd=5)
        #加上預設值，測試方便=============
        self.stockIDentry.grid(row=0, column=1, sticky=tk.W)
        
    #起始日輸入=============
        datelabel=tk.Label(self.stockframe, text="輸入查詢起始日:",font=('Arial',15))
        datelabel.grid(row=1,column=0,sticky=tk.W)

        #加上預設值，測試方便=============
        Date_default = '2022-01-01'
        self.Datevar = tk.StringVar(value=Date_default)
        
        #self.dateentry  = tk.Entry(self.stockframe,text=tk.StringVar(),bd=5) #不要預設值的話改回這行
        self.dateentry  = tk.Entry(self.stockframe,textvariable=self.Datevar,bd=5)
        #加上預設值，測試方便=============
        self.dateentry.grid(row=1, column=1,sticky=tk.W)
    
    #按鈕設定==============
        self.enterFrame = tk.Frame(self.BigFrame1)
        self.enterFrame.pack()

        subminButton1  = tk.Button(self.enterFrame, font=('Microsoft JhengHei',15),text="搜尋",command=self.Calculationstarts)
        subminButton1.grid(row=0, column=0, padx=(0,0))

        subminButton2  = tk.Button(self.enterFrame, font=('Microsoft JhengHei',15),text="下載歷史資料",command=self.DownloadCSV)
        subminButton2.grid(row=0, column=1, padx=(5,0))

        subminButton3  = tk.Button(self.enterFrame, font=('Microsoft JhengHei',15),text="績效回測",command=self.SmaCross1)
        subminButton3.grid(row=0, column=2, padx=(5,0))

        subminButton4  = tk.Button(self.enterFrame, font=('Microsoft JhengHei',15),text="優化策略",command=self.SmaCross2)
        subminButton4.grid(row=0, column=3, padx=(5,0))
        
        subminButton5  = tk.Button(self.enterFrame, font=('Microsoft JhengHei',15),text="清除",command=self.clearData)
        subminButton5.grid(row=0, column=4, padx=(5,0))
        




    #產出視窗==================
        self.outputFrame = tk.Frame(self.BigFrame1)
        self.outputFrame.pack(side=tk.LEFT)

    #公司名稱標籤================
        self.companynameLabel = tk.Label(self.outputFrame, text="股票名稱:", font=("Arial",15))
        self.companynameLabel.grid(row=0, column=0, sticky=tk.W, padx=10,pady=10)
        
        self.companyoutputLabel = tk.Label(self.outputFrame)
        self.companyoutputLabel.configure(text="______________", font=("Arial",15),bg="#F0F0F8")
        self.companyoutputLabel.grid(row=0, column=1, sticky=tk.W, padx=10,pady=10)       
        

    #股價預測標籤================
        self.moneynameLabel = tk.Label(self.outputFrame, text="股價預測:", font=("Arial",15))
        self.moneynameLabel.grid(row=1, column=0, sticky=tk.W, padx=10,pady=10)
        
        self.moneyoutputLabel = tk.Label(self.outputFrame)
        self.moneyoutputLabel.configure(text="______________", font=("Arial",15),bg="#F0F0F8")
        self.moneyoutputLabel.grid(row=1, column=1, sticky=tk.W, padx=10,pady=10)   


#《第二視窗》========================
        self.historyFrame = ttk.LabelFrame(self.BigFrame2,text=f"歷史股價")
        self.historyFrame.pack()

        columns = ('#1','#2', '#3', '#4', '#5', '#6', '#7', '#8', '#9')
        self.tree = ttk.Treeview(self.historyFrame, columns=columns, show='headings')
        self.tree.configure(height=22)
        self.tree.heading('#1', text='日期')
        self.tree.column("#1", minwidth=0, width=80)
        self.tree.heading('#2', text='開盤')
        self.tree.column("#2", minwidth=0, width=48)
        self.tree.heading('#3', text='最高')
        self.tree.column("#3", minwidth=0, width=48)
        self.tree.heading('#4', text='最低')
        self.tree.column("#4", minwidth=0, width=48)
        self.tree.heading('#5', text='收盤')
        self.tree.column("#5", minwidth=0, width=48)
        self.tree.heading('#6', text='還原收盤')
        self.tree.column("#6", minwidth=0, width=55)
        self.tree.heading('#7', text='成交量')
        self.tree.column("#7", minwidth=0, width=85)
        self.tree.heading('#8', text='漲跌幅%')
        self.tree.column("#8", minwidth=0, width=55)
        self.tree.heading('#9', text='BIAS%')
        self.tree.column("#9", minwidth=0, width=48)
        self.tree.pack(side=tk.LEFT)


        #幫treeview加scrollbar------------------------------------------------
        scrollbar = ttk.Scrollbar(self.historyFrame,command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.tree.config(yscrollcommand=scrollbar.set)


#程式撰寫=====================
        
    #參數設定 ==========================
    def Calculationstarts(self):

        sns.set()
        plt.style.use('fivethirtyeight')
        stockCode = self.stockIDentry.get()
        stockNameC = twstock.codes[stockCode].name
        symbol = yf.Ticker(stockCode + ".TW")
        stockName = symbol.info['symbol']                           
        endDate = datetime.datetime.now().date()
        date_str = datetime.datetime(2021,1, 1)
        startDate = datetime.datetime.strptime(self.dateentry.get(), '%Y-%m-%d').date()


        # 下載股價資料======================================
        self.yf_df = yf.download(stockName, startDate, endDate)
        self.yf_df[['Open', 'High', 'Low', 'Close', 'Adj Close']] = self.yf_df[['Open', 'High', 'Low', 'Close', 'Adj Close']].round(2)
        self.yf_df['DailyChange%'] = round((self.yf_df['Close'].pct_change())*100,2)
        self.yf_df['5MA'] = self.yf_df['Close'].rolling(window=5).mean()
        self.yf_df['20MA'] = self.yf_df['Close'].rolling(window=20).mean()
        self.yf_df['BIAS%'] =round((self.yf_df['Close'] - self.yf_df['20MA']) / self.yf_df['20MA'] * 100,2)
        self.yf_df.dropna(inplace=True)

        for index, row in self.yf_df.iterrows():
            date = index.strftime('%Y-%m-%d')
            open_price = str(row['Open'])
            high_price = str(row['High'])
            low_price = str(row['Low'])
            close_price = str(row['Close'])
            adj_close_price = str(row['Adj Close'])
            volume = str(row['Volume'])
            daily_change = str(row['DailyChange%'])
            BIAS=str(row['BIAS%'])
            self.tree_row = (date, open_price, high_price, low_price, close_price, adj_close_price, volume, daily_change,BIAS)
            self.tree.insert("", tk.END, values=self.tree_row)

                


#《第三視窗》========================

        #個股走勢圖 ======================================

        self.fig1 = plt.figure(figsize=(8, 5))
        plt.title(f'{stockCode} PRICE TREND', pad=30, fontsize=20, fontweight='bold',color='#a37e67')
        
        plt.xlabel("Date",fontweight='bold',color='#749cb8', fontsize=13)
        plt.ylabel("Closing Price",fontweight='bold',color='#749cb8', fontsize=13, rotation=0, labelpad=80)
        plt.gca().xaxis.set_label_coords(0.94, -0.07)
        plt.gca().yaxis.set_label_coords(0.06, 1.02)
        plt.plot(self.yf_df["Close"], label='Close', color='navy', linewidth=1.8)
        yf_df2 = yf.download(stockName, startDate, endDate)
        yf_df2['60MA'] = yf_df2['Close'].rolling(window=60).mean()
        plt.plot(self.yf_df['20MA'], label='20MA', linewidth=1.5,color='#e6880e')
        plt.legend()
        
        #改圖片布局================
        plt.tight_layout()
        plt.xticks(fontsize=8.5, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.85)

        #self.fig1 = fig1
        self.fig1_canvas = FigureCanvasTkAgg(self.fig1, self.BigFrame3)
        self.fig1_canvas.get_tk_widget().grid(row=0, column=0, sticky="nw")




#《第四視窗》========================

        #熱力圖(數據集的特徵之間的相關性) ======================================
        self.fig2 = plt.figure(figsize=(8, 7.5))

        plt.title('HeatMap', pad=20, fontsize=20, fontweight='bold',color='#a37e67')        
        sns.set(font_scale=1.2)
        cols = ['Open', 'High', 'Low', 'Close','Adj Close','5MA','20MA', 'Volume']
        self.yf_df_cols = self.yf_df[cols]
        sns.heatmap(self.yf_df_cols.corr(), annot=True,annot_kws={"size": 7})
        plt.xticks(fontsize=9),plt.yticks(fontsize=9)        

        #改圖片布局================
        plt.tight_layout()
        #plt.subplots_adjust(left=0.05, right=1.03, bottom=0.2, top=0.85) 
        plt.subplots_adjust(left=0.07, right=1.03, bottom=0.15, top=0.9) 


        #fig2_canvas = FigureCanvasTkAgg(fig2, self.BigFrame6)
        #fig2_canvas.get_tk_widget().grid(row=0, column=0, sticky="nw")

        self.fig2_canvas = FigureCanvasTkAgg(self.fig2, self.BigFrame4) 
        self.fig2_canvas.get_tk_widget().grid(row=0, column=0, sticky="nw") 



#《第五視窗》========================
        #K線圖 ======================================
        mcolor = mpf.make_marketcolors(up='r', down='g', inherit=True)
        mstyle =mpf.make_mpf_style(base_mpf_style='yahoo',marketcolors=mcolor)   
        title = f'{stockCode}'            
            
        # 繪製 K 線圖
        self.fig3,axes = mpf.plot(self.yf_df, figsize=(8, 6), returnfig=True, type='candle', mav=(5,20,60), volume=True, figratio=(8,6), figscale=1, title=title, style=mstyle, ylabel='Price', xlabel='Date', ylabel_lower='volume', xrotation=360)
        
        #改圖片布局(不知為何無效用)================
        self.fig3.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95, wspace=0)


        self.fig3_canvas = FigureCanvasTkAgg(self.fig3,self.BigFrame5)
        #self.fig3_canvas.draw()
        self.fig3_canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=1)


#《第六視窗》========================
        # 盒鬚圖==============================================================
        self.fig4 = plt.figure(figsize=(8, 6.5))
        sns.boxplot(data=self.yf_df[['Open', 'High', 'Low',  'Close','5MA', '20MA']])
        plt.title('BoxPlot', pad=20, fontsize=20, fontweight='bold',color='#a37e67')

        plt.tight_layout()
        plt.xticks(fontsize=8.5, fontweight='bold')
        plt.yticks(fontsize=10, fontweight='bold')
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.12, top=0.85)        

        self.fig4_canvas = FigureCanvasTkAgg(self.fig4,self.BigFrame6)
        #self.fig4_canvas.draw()
        self.fig4_canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=1)

#《第7視窗》========================
        #散點圖================================================================
        
        self.fig5=sns.pairplot(self.yf_df[['Close', 'Open', 'High', 'Low']],hue="Close",palette="husl").fig        
        plt.title('PairPlot', x=1.25, y=1.9, fontsize=16)

        self.fig5_canvas = FigureCanvasTkAgg(self.fig5, self.BigFrame7)
        self.fig5_canvas.get_tk_widget().config(width=800, height=800)
        self.fig5_canvas.draw()
        self.fig5_canvas.get_tk_widget().pack(side=tk.TOP,fill=tk.BOTH,expand=1)


#《第8視窗》========================


#程式撰寫=====================

        #用決策樹預測股價===================================================
        x = self.yf_df[["Open", "High", "Low", "Volume", "5MA",'20MA']]
        y = self.yf_df["Close"]
        x = x.to_numpy()
        y = y.to_numpy()
        y = y.reshape(-1, 1)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
        model = DecisionTreeRegressor()
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        data = pd.DataFrame(data={"PredictedPrice": ypred})


        # 加入台股邏輯限制 ================================================================

        latest_close_price=self.yf_df['Close'].iloc[-1]
        data["UpperLimit"] = latest_close_price * 1.1
        data["LowerLimit"] = latest_close_price * 0.9
        data.loc[data["PredictedPrice"] > data["UpperLimit"], "PredictedPrice"] = data["UpperLimit"]
        data.loc[data["PredictedPrice"] < data["LowerLimit"], "PredictedPrice"] = data["LowerLimit"]
        tick_size = 0.01
        if latest_close_price < 10:
            tick_size = 0.01
        elif latest_close_price < 50:
            tick_size = 0.05
        elif latest_close_price < 100:
            tick_size = 0.1
        elif latest_close_price < 500:
            tick_size = 0.5
        elif latest_close_price < 1000:
            tick_size = 1
        else:
            tick_size = 5
        data["PredictedPrice_new"] = (data["PredictedPrice"] / tick_size).round() * tick_size
        data["UpperLimit_new"]=round((data["UpperLimit"][0]/tick_size).round() * tick_size,2)
        data["LowerLimit_new"]=round((data["LowerLimit"][0]/tick_size).round() * tick_size,2)
        print('最近收盤價:', latest_close_price, '次日漲停:', data["UpperLimit_new"][0], '次日跌停:', data["LowerLimit_new"][0])
        print(data.loc[:, ['PredictedPrice', 'UpperLimit_new', 'LowerLimit_new', 'PredictedPrice_new']].head(5))

        data['Difference'] = abs(data['PredictedPrice_new'] - latest_close_price)
        sorted_data = data.sort_values('Difference')
        best_prediction = round(sorted_data.iloc[1]['PredictedPrice_new'],2)
        print('最佳預測股價：', best_prediction)



# 結果呈獻=====================

        self.moneyoutputLabel.configure(text=best_prediction)
        self.companyoutputLabel.configure(text=f'{stockCode} ({stockNameC})')

    # 產出 CSV 檔案 ========================================================
    def DownloadCSV(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV files', '*.csv')])
        if file_path:
            #stockCode = self.stockIDentry.get()
            #self.Calculationstarts()
            #self.yf_df.to_csv(file_path + f'./{stockCode}_tw_stock.csv', index=False)
            self.yf_df.to_csv(file_path)


    def SmaCross1(self):
        class SmaCross(Strategy):
            n1=5
            n2=20
            def init(self):
                self.sma1 = self.I(SMA, self.data.Close, self.n1) 
                self.sma2 = self.I(SMA, self.data.Close, self.n2) 
                
            def next(self):
                if crossover(self.sma1, self.sma2): 
                    self.buy()
                elif crossover(self.sma2, self.sma1): 
                    self.sell()

        self.test = Backtest(self.yf_df, SmaCross, cash=1000000, commission=.004,exclusive_orders=True,trade_on_close=True)
        self.result = self.test.run()
            
        self.text1 = tk.Text(self.BigFrame8, font=("Courier", 13), width=100, height=50)
        self.text1.pack(expand=True, fill="both")
        self.text1.insert("end", self.result.to_string(index=True))


#《第9視窗》========================
    def SmaCross2(self):
        self.opt_result_equity = self.test.optimize(n1=range(5, 50, 5),n2=range(10, 120, 5),maximize='Equity Final [$]',constraint=lambda p: p.n1 < p.n2)  

        self.text2 = tk.Text(self.BigFrame9, font=("Courier", 13), width=100, height=50)
        self.text2.pack(expand=True, fill="both")
        self.text2.insert("end", self.opt_result_equity.to_string(index=True))

#《第10視窗》========================
        self.opt_result_sqn = self.test.optimize(n1=range(5, 50, 5),n2=range(10, 120, 5),maximize='SQN',constraint=lambda p: p.n1 < p.n2)  

        self.text3 = tk.Text(self.BigFrame10, font=("Courier", 13), width=100, height=50)
        self.text3.pack(expand=True, fill="both")
        self.text3.insert("end", self.opt_result_sqn.to_string(index=True))


    # 清除資料 ========================================================
    def clearData(self):
        self.moneyoutputLabel.configure(text="______________")
        self.companyoutputLabel.configure(text="______________")
        self.stockIDentry.delete(0, tk.END)
        self.dateentry.delete(0, tk.END)
        if hasattr(self, 'fig1_canvas'): 
            self.fig1.clf() 
            self.fig1_canvas.draw() 
            self.fig1_canvas.get_tk_widget().pack_forget() 
        if hasattr(self, 'fig2_canvas'):
            self.fig2.clf() 
            self.fig2_canvas.draw() 
            self.fig2_canvas.get_tk_widget().pack_forget()           
        if hasattr(self, 'fig3_canvas'):
            #self.fig3.clf() 
            self.fig3_canvas.draw() 
            self.fig3_canvas.get_tk_widget().pack_forget()   
        if hasattr(self, 'fig4_canvas'): 
            #self.fig4.clf() 
            self.fig4_canvas.draw() 
            self.fig4_canvas.get_tk_widget().pack_forget()
        if hasattr(self, 'fig5_canvas'): 
            self.fig5.clf() 
            self.fig5_canvas.draw() 
            self.fig5_canvas.get_tk_widget().pack_forget()
        if hasattr(self, 'text1'):
            self.text1.delete("1.0", "end")
            self.text1.destroy() 
        if hasattr(self, 'result'):
            del self.result
        if hasattr(self, 'text2'):
            self.text2.delete("1.0", "end")
            self.text2.destroy() 
        if hasattr(self, 'text3'):
            self.text3.delete("1.0", "end")
            self.text3.destroy() 
        for row_id in self.tree.get_children():
            self.tree.delete(row_id)        
        







    



#結束===================================
def main():
    window = Window()
    window.title("台股價格預測查詢系統")
    imge = tk.PhotoImage(file='icon2.png')
    window.iconphoto(False,imge)
    window.mainloop()

if __name__ == "__main__":
    main()
