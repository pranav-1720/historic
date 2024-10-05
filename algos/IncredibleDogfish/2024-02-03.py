from typing import Any, Optional, Tuple, Union
import uuid
import numpy as np
import pandas as pd
import pickle
import torch
from scipy.stats import norm
import torch.nn.functional as F

class GaussianProcess:
    def __init__(self, kernel, noise=1e-6):
        self.kernel = kernel
        self.noise = noise

    def fit(self, X_train, y_train):

        K = self.kernel(X_train, X_train) + (self.noise + 1e-3) * torch.eye(len(X_train))
        self.X_train = X_train
        try:
            self.L = torch.linalg.cholesky(K)
        except torch._C._LinAlgError:
            self.L = None  # No Cholesky factor available
            self.alpha = torch.pinverse(K) @ y_train
            return
        self.alpha = torch.cholesky_solve(y_train, self.L)

    def predict(self, X_test):
        X_test = X_test.to(self.X_train.dtype)
        K_trans = self.kernel(self.X_train, X_test)
        mu = K_trans.T @ self.alpha
        v = torch.cholesky_solve(K_trans, self.L)
        var = self.kernel(X_test, X_test) - K_trans.T @ v
        
        if self.L is not None:
            mu = K_trans.T @ self.alpha
            v = torch.cholesky_solve(K_trans, self.L)
            var = self.kernel(X_test, X_test) - K_trans.T @ v
        else:
            mu = K_trans.T @ self.alpha
            K_inv = torch.pinverse(self.kernel(self.X_train, self.X_train) + (self.noise + 1e-3) * torch.eye(len(self.X_train)))
            var = self.kernel(X_test, X_test) - K_trans.T @ K_inv @ K_trans
        return mu, var

class RBFKernel:
    def __init__(self, length_scale=1.0):
        self.length_scale = length_scale

    def __call__(self, X1, X2):
        sqdist = torch.cdist(X1, X2, p=2).pow(2)
        return torch.exp(-0.5 * sqdist / self.length_scale**2)


def bayesian_optimization_step(gp,bounds):
    pri_tensor = np.random.uniform(bounds[0], bounds[1], 100)
    pri_tensor = torch.tensor(pri_tensor)
    X = pri_tensor
    X = X.reshape(-1, 1)

    test_x = torch.cat([X], dim=-1)
    with torch.no_grad():
        mu, var = gp.predict(test_x)
    sigma = torch.sqrt(torch.diag(var)).unsqueeze(1)
    
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.nan_to_num(sigma, nan=0.0, posinf=0.0, neginf=0.0)
    
    max_revenue = torch.max(mu)
    target_revenue_threshold = max_revenue * 0.9
    imp = mu - target_revenue_threshold - 0.01
    Z = imp / (sigma + 1e-9)
    ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
    return test_x[torch.argmax(ei), 0].item()  
    within_threshold = mu >= target_revenue_threshold
    
    if within_threshold.any():
        filtered_prices = X[within_threshold]
        
        max_price_index = torch.argmax(filtered_prices)
        best_price = filtered_prices[max_price_index].item()
    else:
        best_price = X[torch.argmax(mu)].item()

    return best_price 


def p(
    current_selling_season: int,
    selling_period_in_current_season: int,
    prices_historical_in_current_season: Union[np.ndarray, None],
    demand_historical_in_current_season: Union[np.ndarray, None],
    competitor_has_capacity_current_period_in_current_season: bool,
    information_dump=Optional[Any],
) -> Tuple[float, Any]:
    """
    MAIN PRICING FUNCTION WHICH IS REQUIRED IN ANY SUBMISSION

    Return the price to set for the next period.

    ARGUMENTS
    ----------
    current_selling_season : int
            The current selling season (1, 2, 3, ..., 100).
    selling_period_in_current_season : int
            The period in the current season (1, 2, ..., 100).
    prices_historical_in_current_season : Union[np.ndarray, None]
            A two-dimensional array of historical prices: 
                rows index the competitors
                columns index the historical selling periods. 
            Equal to `None` if `selling_period_in_current_season == 1`.
    demand_historical_in_current_season : Union[np.ndarray, None]
            A one-dimensional array of historical (own) demand. 
            Equal to `None` if `selling_period_in_current_season == 1`.
    competitor_has_capacity_current_period_in_current_season : bool
            `False` if competitor is out of stock, else `False`
    information_dump : Any, optional, default `None`
            Custom object to pass information from one selling_period to the other.
            Equal to `None` if `selling_period_in_current_season == 1`.
            To pass information from one selling_season to the other, 
            use the 'duopoly_feedback.data' file.

    RETURNS
    -------
    Tuple[float, Any]
        The price and a the information_dump (with, e.g., a state of the model).


    Examples
    --------

    >>> prices_historical_in_current_season.shape == (2, selling_period_in_current_season - 1)
    True

    >>> demand_historical_in_current_season.shape == (selling_period_in_current_season - 1, )
    True

    Hints
    ----------
    To pass to yourself information for download in the DPC website, use `duopoly_feedback.data`.
    This file will be available in the website under `Download Feedback Data` in the results page.

    """
    # first set some shorter names for convenience
    day = selling_period_in_current_season
    season = current_selling_season
    demand = demand_historical_in_current_season
    prices = prices_historical_in_current_season

    # set some indicator variables
    first_day = True if day == 1 else False
    last_day = True if day == 100 else False
    first_season = True if season == 1 else False
    season_previous_period = season if not first_day else season - 1
    
    if(season>=20 and day==1):
        # every seasons first value of price is x_train
        X_train = torch.tensor(information_dump['history'][information_dump['history']['day'] == 1]['own_price'].values.reshape(-1, 1))
        y_train = torch.tensor(information_dump['history'][information_dump['history']['day'] == 99]['cumulative_revenue'].values.reshape(-1, 1))
        # if season %20 ==0:
            # print(X_train.shape)
            # print(y_train.shape)
            # print(X_train)
            # print(y_train)
        kernel = RBFKernel(length_scale=1.0)
        gp = GaussianProcess(kernel)
        gp.fit(X_train, y_train)
        information_dump['gp'] = gp
    if first_day:
        if season < 20:
            # print('Randomizing')
            # print('Season:', season)
            # print('Day:', day)
            price = np.random.randint(30, 100)
            print(price)
        else:
            gp = information_dump['gp']
            price = bayesian_optimization_step(gp, [30, 100])

        
    elif 1< day <= 5:
        # Randomize in the first period of the season
        price = prices[0][-1]
    else:
        previous_demand = demand[day-6:day-1]
        average_demand = np.mean(previous_demand)
        inventory = 80-demand.sum()         
        if ((100-day)*average_demand < inventory):
            price = prices[0][-1]-1
        else:
            price = prices[0][-1]+1
        # Use the average_demand for further calculations
        # ...
        # if first day of simulation, initialize information dump from feedback data object
    
    if first_day:
        if first_season:
            information_dump = _initialize_data_feedback()
            # we just started a new simulation, let's give it an ID
            my_simulation_id = "%s" % uuid.uuid4()
            information_dump['current_simulation'] = my_simulation_id
        # initialize the
        information_dump['cumulative_revenue_current_selling_season'] = 0
    if not first_day:
        # do the book keeping (unless we are on the first day)
        remaining_capacity = 80 - np.sum(demand)
        revenue = demand[-1] * prices[0, -1]
        my_simulation_id = information_dump['current_simulation']
        rev_sofar_current_selling_season = information_dump[
            'cumulative_revenue_current_selling_season']
        # append and update info in information_dump object
        if(day == 100):
            print('Season:', season)
            print('Day:', day)
            print('total revenue:', rev_sofar_current_selling_season+revenue)
        new_data ={
                    'simulation': my_simulation_id,
                    'day': day - 1,
                    'season': season_previous_period,
                    'demand': demand[-1],
                    'own_price': prices[0, -1],
                    'competitor_price': prices[1, -1],
                    'remaining_capacity': remaining_capacity,
                    'revenue': revenue,
                    'cumulative_revenue': rev_sofar_current_selling_season+revenue,
        }   
        information_dump['history'] = pd.concat([information_dump['history'], pd.DataFrame([new_data])], ignore_index=True)
        information_dump['cumulative_revenue_current_selling_season'] = rev_sofar_current_selling_season+revenue
    return price, information_dump

    # Save information dump to disk (duopoly_feedback.data) in selling period 100 & 101
    # (Note, 101 is only happening under the hood in the simulation to obtain the demand of selling period 100.)
    if selling_period_in_current_season >= 100:
        with open('duopoly_feedback.data', 'wb') as handle:
            pickle.dump(information_dump, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    return price, information_dump


def _initialize_data_feedback():
    """Initialize the feedback data object"""

    # Try to load duopoly feedback data from disk
    # try:
    #     with open('duopoly_feedback_gp.data', 'rb') as handle:
    #         feedback = pickle.load(handle)
    #     return feedback
    # except:
    return {

            # keep track of historical observables
            'history': (
                pd.DataFrame({
                    'simulation': '',
                    'day': [],
                    'season': [],
                    'demand': [],
                    'own_price': [],
                    'competitor_price': [],
                    'remaining_capacity': [],
                    'revenue': [],
                    'cumulative_revenue': [],
                })
            ),
            'current_simulation': '',
            'cumulative_revenue_current_selling_season': 0,
            'gp': None
        }
