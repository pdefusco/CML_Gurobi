import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split      #importing scikit-learn's function for data splitting
from sklearn.ensemble import GradientBoostingRegressor    #importing scikit-learn's gradient booster regressor function
from sklearn.model_selection import cross_validate        #improting scikit-learn's cross validation function
import cml.data_v1 as cmldata
import cdsw


@cdsw.model_metrics
def predict(args):
    """
    Method to predict optimal Price for the Two Categories, Number of Items per Category, and Total Revenue
    """

    df = pd.DataFrame(data=args, dtype=np.float64)

    X = df[["p[1]","p[2]"]]
    y = df["n[1]"]

    # Split the data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.75, random_state=1
    )

    linear_regressor = make_pipeline(LinearRegression())
    linear_regressor.fit(X_train, y_train)
    linear_regression_validation = cross_validate(linear_regressor, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

    linear_regression_validation['train_score'],linear_regression_validation['test_score']

    from sklearn.ensemble import GradientBoostingRegressor
    xgb_regressor = make_pipeline(GradientBoostingRegressor(n_estimators=10))
    xgb_regressor.fit(X_train, y_train)
    xgb_regressor_validation = cross_validate(xgb_regressor, X_train, y_train, cv=5, return_train_score=True, return_estimator=True)

    xgb_regressor_validation['train_score'], xgb_regressor_validation['test_score']

    #### Initialize the model
    m = gp.Model("price optimization")

    products = [1,2]            #### Category 1 and Category 2
    N = 200                     #### limit on available space
    l = 0                       #### price control, we'll start this at 0

    p = m.addVars(products, name="p")            #### price decision variables
    n = m.addVars(products, name="n")            #### decision variable for number of items in each category

    min_items = {1:50,2:50}
    price_bounds = {1:[300,400], 2:[100,300]}
    m.addConstrs(n[c] >= min_items[c] for c in products)        #### we could hardcode 50 instead of min_items, but this is more flexible
    m.addConstr(p[1] == [300,400])                              #### this is a shorthand way to code 300 <= p[1] <= 400
    m.addConstr(p[2] == [100,300]);
    m.addConstr(n.sum() ==  N);     #### remember we set N = 200 earlier
    m.addConstr(p[1]-p[2] == [50,100]);

    revenue = gp.quicksum(p[c]*n[c] for c in products)          #### you could also use the more simple p.prod(n)
    penalty = l*(p[1]**2+p[2]**2)                               #### we used l as the lambda parameter earlier
    m.setObjective(revenue - penalty, sense = GRB.MAXIMIZE)

    #### install the package and load the required function
    from gurobi_ml import add_predictor_constr

    m_feats = pd.DataFrame({"p[1]":[p[1]],"p[2]":[p[2]]})

    pred_constr = add_predictor_constr(m, xgb_regressor, m_feats, n[1])
    pred_constr.print_stats()

    m.Params.NonConvex = 2
    m.optimize()

    # Track inputs
    cdsw.track_metric("input_data", args)

    print("\nOptimal price for the two categories:\n",round(p[1].X,2),round(p[2].X,2))
    print("\nOptimal number of space assigned to the two categories:\n",round(n[1].X), round(n[2].X))
    print("\nTotal revenue:\n",round(revenue.getValue(),2))

    return {"data": dict(args), "optimal prices": tuple([round(p[1].X,2),round(p[2].X,2)]), "optimal product quantities": tuple([round(n[1].X), round(n[2].X)]), "total revenue": round(revenue.getValue(),2)}
