import pandas as pd
import numpy as np
import networkx
import matplotlib.pyplot as plt
import itertools
from collections import Counter
from scipy import linalg
import scipy
from scipy import stats

import seaborn as sns
import numpy as np
import tabulate
import sklearn.utils
import random
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

class WineRanking(object):
    def __init__(self):
        """
        INPUT:
            csv_path (str) - path to csv file from which to load data.
            save file as results.csv in working directory with following headers 
                
                [u'user', u'wine', u'Rank', u'Appearance', u'Aroma', u'Body', u'Taste',
                 u'Finish',u'Region', u'Varietal', u'Vintage',u'Price']
        OUTPUT:
            Instantiates WineRanker Object.
            
        
        """
        self.data = pd.read_csv('ranking.csv')
        try: 
            self.data[[u'user', u'wine', u'Rank', u'Appearance', u'Aroma',
                u'Body', u'Taste',u'Finish',u'Region', u'Varietal',
                u'Vintage',u'Price']]
        except KeyError as e: 
            print e
        self.wine_name_dict = dict(zip(pd.read_csv('wines.csv')['wine_id'],pd.read_csv('wines.csv')['name'] ))
        self.wine_info = pd.read_csv('wines.csv')
        self.data['Wine'] = self.data['wine'].map(self.wine_name_dict)
        self.n_judges, self.n_wines = self.data.set_index(['user','wine'])['Rank'].unstack().shape
    def summary_table(self, how='Rank'):
        self.sum_table = self.data.groupby('wine')['Rank'].agg([{'Average' : lambda x: np.round(np.mean(x),1),
                                                     'Median' : np.median,
                                                     'Sum': np.sum,
                                                     'Scores': lambda x: [i for i in x]}]).T.reset_index(level=[0] ,drop=True).T.reset_index()
        self.sum_table['Wine'] = self.sum_table['wine'].map(self.wine_name_dict)
        self.sum_table = self.sum_table.set_index(['Wine', 'wine']).sort_values('Average')
        self.sum_table['OrdinalRank'] = scipy.stats.rankdata(self.sum_table['Average'], method='ordinal')
        return self.sum_table
################        
    def winners(self):
        display = self.summary_table().reset_index().sort_values('Average')
        display['Placement'] = ['1st','2nd','3rd','4th', '5th', '6th']
        display = display.rename(columns={'wine':'Label'})
        return display[['Placement','Label','Wine']].set_index('Placement')
    def winners_details(self):
        display = self.summary_table().reset_index().sort_values('Average')
        display['Placement'] = ['1st','2nd','3rd','4th', '5th', '6th']
        display = display.rename(columns={'wine':'Label'})
        return display.set_index('Placement')
    def unveil_wines(self, report=True):
        
        display = self.summary_table().reset_index().sort_values('Average')
        self.ranked = display['wine']
        
        
        if report:
            print '*'*120, '\n\n','\t'*6,'Flight 1:\n','\t'*6,'Number of Judges = ',self.n_judges,'\n','\t'*6,'Number of Wines: ',self.n_wines
            print  '\n\n', '*'*120
            print "Identification of the Wine:\t\tThe judges' overall ranking based on Rank"
            for i in zip(display.T, ['1st','2nd','3rd','4th', '5th', '6th']):
                print '  ', display.ix[i[0]]['wine'], ' is ', display.ix[i[0]]['Wine'][:5] ,'\t\t\t .........  ',  i[1]+' Place'
        else: 
            return display
    def judges_scores(self, report=True):
        self.scores = self.data.set_index(['user','wine']).unstack()['Rank']
        if report:
            print "\n\n\t\t\tThe Judge's Rankings:\n\n", tabulate.tabulate(self.scores, headers=self.scores.columns,  tablefmt="simple"),'\n\n\n\n'
        else:
            return self.scores
        
    
    def votes_against(self, report=True):
        try:
            self.ranked
        except AttributeError:
            self.unveil_wines(report=False)
        votes_against =  self.judges_scores(report=False)[list(self.ranked)].sum().reset_index().rename(columns={0:'Votes Against'}).merge(self.ranked.reset_index())
        votes_against['index'] = votes_against['index']+1
        votes_against = votes_against.rename(columns={'index':'Group Rank'}).set_index('wine').T
        if report: 
            print "\n\n\t\t\tTable of Votes Against:\n\n", tabulate.tabulate(votes_against, headers=votes_against.columns,  tablefmt="simple"),'\n\n'
            print '\t\t\t(',self.n_judges,' is the best possible, ',self.n_judges*self.n_judges, 'is the worst)\n\n\n\n'
        else: return votes_against
    
    def judges_vs_average(self, report=True):
        if report:
            print 'Here is a measure of the correlation in the preferences of the judges \nwhich ranges between 1.0 (perfect correlation) and 0.0 (no correlation):\n'
            F, Fnorm, pvalue = self.FriedmanChiSquare()
            print '\t\t\t W = ', np.round(Fnorm, 2), '\n\n'
            print 'The probability that random chance could be responsible for this correlation is',pvalue
            if pvalue > .1:
                print 'Since the p-value is above 0.1 the judges preferences are not strongly related'
            else: 
                print 'Since the p-value is below 0.1 the judges preferences are strongly related'
            print "\t\t\tFriedman = ", np.round(F, 2), '\n\n'
            print '\nFinally, when n or k is large (i.e. n > 15 or k > 4), the probability distribution of Q can be approximated by that of a chi-squared distribution. In this case the p-value is given by {\displaystyle \mathbf {P} (\chi _{k-1}^{2}\geq Q)} \mathbf{P}(\chi^2_{k-1} \ge Q). If n or k is small, the approximation to chi-square becomes poor and the p-value should be obtained from tables of Q specially prepared for the Friedman test. If the p-value is significant, appropriate post-hoc multiple comparisons tests would be performed.'
        else:
            return F, Fnorm, pvalue
    
    
    def FriedmanChiSquare(self, how='Rank'):
        
        ranks = [list(group[how]) for i, group in self.data[['wine',how]].groupby('wine')]
        if self.n_wines == 6: 
            F,pvalue = scipy.stats.friedmanchisquare(ranks[0], ranks[1], ranks[2], ranks[3], ranks[4], ranks[5])
        if self.n_wines == 8: 
            F,pvalue = scipy.stats.friedmanchisquare(ranks[0], ranks[1], ranks[2], ranks[3], ranks[4], ranks[5], ranks[6], ranks[7])
        Fnorm = F/(self.n_judges*(self.n_wines - 1))
        return F, Fnorm, pvalue
    
    
    def rank_correlation(self, how='Rank', report=True):
        
        corr_results = pd.DataFrame()
        for i, group in self.data.groupby('user'):
            #print group[['Rank','wine']]
            corr_rank = group.merge(self.summary_table().reset_index()[['wine','OrdinalRank']])[['OrdinalRank','wine','Rank']]
            kendalltau, pvalue = scipy.stats.kendalltau(corr_rank['OrdinalRank'],corr_rank['Rank'])
            row = {'user':i,'Correspondance': np.round(kendalltau,2), 'p-value':pvalue}
            corr_results = corr_results.append(row, ignore_index=True)
        corr_results = corr_results.set_index('user')
        if report: 
            print '\nContrary to the Spearman correlation, the Kendall correlation is not affected by\n how far from each other ranks are but only by whether the ranks between observations\nare equal or not, and is thus only appropriate for discrete\nvariables but not defined for continuous variables.'    
            print '\n\n\n\n\t  Correlation Between the Ranks of \nEach Person With the Average Ranking of Others'
            print '\n\nValues close to 1 indicate strong agreement, values close to -1 indicate strong disagreement.\n'
            print tabulate.tabulate(corr_results, corr_results.columns, tablefmt='simple')
        
        else:
            return corr_results
    
    def pairwise_correlation(self, how='Rank', report=True):
        r_correlation = pd.DataFrame()
        for c in itertools.combinations(self.data.user.unique(),2):
            x = self.data[self.data.user == c[0]][how]
            y = self.data[self.data.user == c[1]][how]
            correlation1, pvalue_1 = scipy.stats.kendalltau(x,y)  
            correlation2, pvalue_2 = scipy.stats.spearmanr(x,y)
            correlation3, pvalue_3 = scipy.stats.pearsonr(x,y)
            row = {'judge1':c[0], 'judge2':c[1],
                   'KendallTau': correlation1, 'pvalue_KT':pvalue_1,
                   'SpearmanR': correlation2, 'pvalue_SR':pvalue_2,
                   'PearsonR': correlation3, 'pvalue_PR':pvalue_3  }
            r_correlation = r_correlation.append(row, ignore_index=True)
       
       
        pairwise = r_correlation
        pairwise['significant'] = None
        pairwise['significant'][pairwise['pvalue_KT'] < 0.05] = 'Significant'
        pairwise = pairwise[['judge1','judge2','KendallTau','pvalue_KT','significant']].sort_values(['significant','KendallTau'], ascending=False)
        if report:     
            print '\n\n\n\n\t Pairwise Rank Correlations'
            print tabulate.tabulate(pairwise, pairwise.columns, tablefmt='simple')
        else:
            pairwise = pairwise.set_index(['judge1', 'judge2'])
            return pairwise, r_correlation
    def rank_wines_pdf_report(self):
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("template/wineranker.html")
        template_vars = {"title" : "Liquid Assets",
                 'n_judges':self.n_judges,
                 'n_wines': self.n_wines,
                 "winners": self.winners().to_html(),
                'title2': 'Details: some summary data',
                'title_d': 'Descriptive Details',
                'wine_des':self.wine_info[['name','Region','Varietal','Year','Price']].merge(self.winners_details().rename(columns={'Wine':'name'}).drop(['Median', 'Sum','Scores'], axis=1), on='name').to_html(),
                'winners_details':self.winners_details().to_html(),
                'title3':"The Judge's Rankings:",
                 'judges_rankings':self.judges_scores(report=False).to_html(),
                 'title4':"Table of Votes Against:",
                 'votes_against':self.votes_against(report=False).to_html(),
                 'high_n_judges': self.n_judges*self.n_judges,
                'title5':'Rank Correlation',
                 'Friedman':self.FriedmanChiSquare()[0],
                'F_norm':self.FriedmanChiSquare()[1],
                'f_pvalue':self.FriedmanChiSquare()[2],
                 'title6':'Correlation Between the Ranks',
                 'title7':'of Each Person With the Average Ranking of Others',
                 'rank_correlation':self.rank_correlation(report=False).sort_values('Correspondance', ascending=False).to_html(),
                 'title8':'Pairwise Rank Correlations',
                 'pairwise_ranks':self.pairwise_correlation(report=False)[0].sort_index().to_html(),
                'title9':'Feature Correlation Plot',
                 'title10':'HeatMap',
                 'title11':'Feature Correlation KDE plot',
                'title12':'Pairwise Correlation Heat Map',
                'title13':'Simulated Results', 
                'simulated_scores':self.bootstrap(self.n_judges)[0][['Average','Median', 'Sum']].to_html()
                         
                 }
        html_out = template.render(template_vars)
        HTML(string=html_out).write_pdf('WineRankerReport.pdf', stylesheets=["template/typography.css"])
    
    
    
    
    def rank_wines_report(self, groupby='wine', how='Rank'):
        self.unveil_wines()
        self.judges_scores()
        self.votes_against()
        self.judges_vs_average()
        self.rank_correlation()
        self.pairwise_correlation()
    
    def run_all(self):
        self.all_rankings = pd.DataFrame() 
        for i in ['Rank', u'Appearance', u'Aroma', u'Body', u'Taste', u'Finish','Price']:
            temp_df = self.rank_wines(groupby= 'wine', how=i)
            temp_df['Feature'] = i
            self.all_rankings = self.all_rankings.append(temp_df)
    def generate_awesome_plots(self):
        sns.set_context('poster')
        sns.set_style("dark", {'axes.grid' : False})
        sns.pairplot(self.data, kind='reg')
        sns.plt.savefig('images/Corrplot.png', dpi=300)
        sns.plt.close()
        g = sns.PairGrid(self.data)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)
        sns.plt.grid('off')
        sns.plt.savefig('images/Corrplot_kde.png', dpi=300) 
        sns.plt.close()
        corr = self.data.corr()
        sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
        sns.plt.savefig('images/Corrplot_heat.png', dpi=300)
        sns.plt.close()
        sns.heatmap(self.pairwise_correlation(report=False)[1].set_index(['judge1','judge2']).sort_index(axis=1)['KendallTau'].unstack().T,cmap=plt.cm.seismic)
        sns.plt.savefig('images/Pairwise_ranking.png', dpi=300)
        sns.plt.close()

    def bootstrap(self, n_sample):
        self.simulated = pd.DataFrame()
        for i in np.arange(0,n_sample, 1):
            result = pd.DataFrame(random.sample(np.arange(1,self.n_wines+1,1), self.n_wines))
            self.simulated = self.simulated.append(result.T)
        self.simulated.columns = self.summary_table().reset_index().sort_values('wine').wine
        self.sim_data = pd.DataFrame(self.simulated.reset_index().drop('index', axis=1).unstack().reset_index().rename(columns={'level_1':'user', 0:'Rank'}))

        
        self.sim_all_scores = self.sim_data.set_index(['user','wine'])['Rank'].unstack()
        self.sim_n_judges, self.sim_n_wines = self.sim_all_scores.shape
        self.sim_sum_table = self.sim_data.groupby('wine')['Rank'].agg([{'Average' : lambda x: np.round(np.mean(x),1),
                                                     'Median' : np.median,
                                                     'Sum': np.sum,
                                                     'Scores': lambda x: [i for i in x]}]).T.reset_index(level=[0] ,drop=True).T
        self.sim_sum_table = self.sim_sum_table.reset_index()
        self.sim_sum_table['Wine'] = self.sim_sum_table['wine'].map(self.wine_name_dict)
        self.sim_sum_table = self.sim_sum_table.set_index(['Wine', 'wine']).sort_values('Average') 
        
        stacked = pd.DataFrame(self.sim_all_scores.stack()).reset_index()
        r_correlation = pd.DataFrame()
        for c in itertools.combinations(stacked.user.unique(),2):
            x = stacked[stacked.user == c[0]][0]
            y = stacked[stacked.user == c[1]][0]
            correlation1, pvalue_1 = scipy.stats.kendalltau(x,y)  
            correlation2, pvalue_2 = scipy.stats.spearmanr(x,y)
            correlation3, pvalue_3 = scipy.stats.pearsonr(x,y)
            row = {'judge1':c[0], 'judge2':c[1],
                   'KendallTau': correlation1, 'pvalue_KT':pvalue_1,
                   'SpearmanR': correlation2, 'pvalue_SR':pvalue_2,
                   'PearsonR': correlation3, 'pvalue_PR':pvalue_3  }
            r_correlation = r_correlation.append(row, ignore_index=True)
       
       
        pairwise = r_correlation
        pairwise['significant'] = None
        pairwise['significant'][pairwise['pvalue_KT'] < 0.05] = 'Significant'
        pairwise = pairwise[['judge1','judge2','KendallTau','pvalue_KT','significant']].sort_values(['significant','KendallTau'], ascending=False)
        pairwise = pairwise.set_index(['judge1', 'judge2'])
        sns.heatmap(pairwise['KendallTau'].unstack().T,cmap=plt.cm.seismic)
        sns.plt.savefig('images/Pairwise_ranking_bootstrap.png', dpi=300)
        sns.plt.close()
        return self.sim_sum_table, self.sim_all_scores, pairwise

        