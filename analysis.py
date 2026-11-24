"""Reproduce the individual regression assignment. No auction outcomes enter fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, root_mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
COST = 1500
SEED = 42
FEATURES = ['Mileage_K', 'Age_Years', 'Horsepower', 'Accident', 'Num_Owners']

def encode(frame):
    x = frame[FEATURES].copy()
    if not x['Accident'].isin(['yes', 'no']).all():
        raise ValueError('Accident must be yes or no')
    x['Accident'] = x['Accident'].map({'yes': 1, 'no': 0})
    return x.astype(float)

def buy_or_pass(predicted_sale_price, auction_price):
    """Buy only if predicted retail proceeds strictly exceed all-in cost."""
    return 'buy' if auction_price + COST < predicted_sale_price else 'pass'

def dealer_decision(car, model, features):
    """Take a new car's five features and Auction_Price, return buy/pass and prices."""
    x = encode(pd.DataFrame([car]))[features]
    x = sm.add_constant(x, has_constant='add')
    prediction = float(model.predict(x).iloc[0])
    return {'predicted_sale_price': prediction,
            'maximum_break_even_bid': prediction - COST,
            'decision': buy_or_pass(prediction, float(car['Auction_Price']))}

def main():
    output = ROOT/'results'; output.mkdir(exist_ok=True)
    cars = pd.read_csv(ROOT/'data/used_car_sales.csv')
    lots = pd.read_csv(ROOT/'data/auction_lots.csv')
    assert len(cars) == 240 and len(lots) == 40
    assert not cars.isna().any().any() and not lots.isna().any().any()
    train, test = train_test_split(cars, test_size=0.30, random_state=SEED)
    assert len(train)==168 and len(test)==72 and set(train.index).isdisjoint(test.index)
    full = sm.OLS(train.Sale_Price, sm.add_constant(encode(train))).fit()
    selected = FEATURES.copy()
    model = full
    reason = 'Retained all five features.'
    # One prespecified comparison: remove owners only if training evidence supports it.
    reduced_features = [f for f in FEATURES if f != 'Num_Owners']
    reduced = sm.OLS(train.Sale_Price, sm.add_constant(encode(train)[reduced_features])).fit()
    if full.pvalues['Num_Owners'] > .05 and reduced.rsquared_adj > full.rsquared_adj:
        selected = reduced_features; model = reduced
        reason = ('Dropped Num_Owners: full-model p-value exceeds 0.05 and removing it '
                  'improves training adjusted R-squared. No test data used for selection.')
    mean = float(train.Sale_Price.mean())
    baseline = np.full(len(test), mean)
    predicted = model.predict(sm.add_constant(encode(test)[selected]))
    metrics = {}
    for name, pred in [('baseline', baseline), ('v1', predicted)]:
        metrics[name] = {'test_r2': float(r2_score(test.Sale_Price,pred)),
                         'test_rmse': float(root_mean_squared_error(test.Sale_Price,pred))}
    lot_predictions = model.predict(sm.add_constant(encode(lots)[selected]))
    evaluated = lots.copy()
    evaluated['baseline_prediction'] = mean
    evaluated['v1_prediction'] = lot_predictions
    evaluated['profit_if_bought'] = evaluated.Later_Sale_Price - evaluated.Auction_Price - COST
    for name in ['baseline','v1']:
        buy = evaluated.apply(lambda r: buy_or_pass(r[name+'_prediction'],r.Auction_Price)=='buy',axis=1)
        evaluated[name+'_buy'] = buy
        metrics[name]['cars_bought'] = int(buy.sum())
        metrics[name]['losing_cars'] = int((buy & (evaluated.profit_if_bought<0)).sum())
        metrics[name]['auction_profit'] = float(evaluated.loc[buy,'profit_if_bought'].sum())
    example = lots.iloc[0].drop(labels=['Later_Sale_Price']).to_dict()
    decision = dealer_decision(example,model,selected)
    decision['baseline_decision'] = buy_or_pass(mean,float(example['Auction_Price']))
    decision['baseline_prediction'] = mean
    report = {
        'data': {'source':'Instructor-provided used_car_sales.csv and auction_lots.csv',
                 'historical_period':'Last year, as described in the prompt; no exact dates supplied',
                 'historical_rows':240,'training_rows':168,'test_rows':72,'auction_rows':40,
                 'random_seed':SEED,'readiness_cost':COST},
        'outcome':'Sale_Price (USD)','selected_features':selected,'selection_reason':reason,
        'training_adjusted_r2':float(model.rsquared_adj),
        'full_training_adjusted_r2':float(full.rsquared_adj),
        'reduced_training_adjusted_r2':float(reduced.rsquared_adj),
        'full_owners_p_value':float(full.pvalues['Num_Owners']),
        'coefficients':{k:float(v) for k,v in model.params.items()},
        'p_values':{k:float(v) for k,v in model.pvalues.items()},
        'metrics':metrics,'new_car':example,'new_car_decision':decision,
        'incremental_profit':metrics['v1']['auction_profit']-metrics['baseline']['auction_profit'],
        'rmse_reduction_pct':100*(1-metrics['v1']['test_rmse']/metrics['baseline']['test_rmse'])
    }
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'ols_summary.txt').write_text(str(model.summary())+'\n\nFull model\n'+str(full.summary()))
    evaluated.to_csv(output/'auction_decisions.csv',index=False)
    pd.DataFrame({'source_row':test.index+2,'actual':test.Sale_Price,
                  'baseline':baseline,'v1':predicted}).to_csv(output/'test_predictions.csv',index=False)
    pd.DataFrame({'source_row':train.index+2}).to_csv(output/'training_rows.csv',index=False)

    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig, ax = plt.subplots(figsize=(7.2,4.3),layout='constrained')
    actual = test.Sale_Price.to_numpy()/1000
    ax.scatter(actual,baseline/1000,label='Baseline',color='#9ca3af',s=25,alpha=.65)
    ax.scatter(actual,np.asarray(predicted)/1000,label='Regression v1',color='#176b87',s=30,alpha=.8)
    lim=[min(actual.min(),np.min(predicted)/1000)-2,max(actual.max(),np.max(predicted)/1000)+2]
    ax.plot(lim,lim,color='#30343b',linewidth=1,linestyle='--',label='Perfect prediction')
    ax.set(xlabel='Actual retail sale price ($ thousands)',ylabel='Predicted price ($ thousands)',xlim=lim,ylim=lim)
    ax.legend(frameon=False,fontsize=10)
    fig.savefig(output/'predicted_vs_actual.png',dpi=180);plt.close(fig)
    assert buy_or_pass(20000,18500)=='pass'
    assert buy_or_pass(20000,18499)=='buy'
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
