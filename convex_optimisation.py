import os
import json

import numpy as np
from scipy.optimize import minimize


class Optimisation:
    def __init__(self,ingredients_name,ingredients_quantity,constraints):
        MACRO_NUTRIENTS={}
        with open("dataset_nutrients.json") as f:
            MACRO_NUTRIENTS=json.load(f)
            
        self.constraints=constraints
        self.ratios=[i/ingredients_quantity[0] for i in ingredients_quantity]


        self.calories_per_gram=[]
        self.proteins_per_gram=[]
        self.fats_per_gram=[]


        for i in ingredients_name:
            try:
                temp=MACRO_NUTRIENTS[i]
                self.calories_per_gram.append(MACRO_NUTRIENTS[i]['calories'])
                self.proteins_per_gram.append(MACRO_NUTRIENTS[i]['protein'])
                self.fats_per_gram.append(MACRO_NUTRIENTS[i]['fats'])
            except KeyError:
                self.calories_per_gram.append(0)
                self.proteins_per_gram.append(0)
                self.fats_per_gram.append(0)
                
                
        
        self.calories_per_gram=np.array(self.calories_per_gram)
        self.proteins_per_gram=np.array(self.proteins_per_gram)
        self.fats_per_gram=np.array(self.fats_per_gram)
        
        pass

    def constraint_func(self,x):
        base = x[0] / self.ratios[0]
        return [(x[i] - self.ratios[i] * base) for i in range(1, len(x))]

    def objective(self,x):
        total_calories = np.dot(x, self.calories_per_gram)
        total_proteins = np.dot(x, self.proteins_per_gram)
        total_fats = np.dot(x, self.fats_per_gram)

        min_calories=-1
        min_proteins=-1
        min_fats=-1
        
        max_calories=1e8
        max_proteins=1e8
        max_fats=1e8
    
        if("calories" in self.constraints.keys()):
            min_calories=self.constraints["calories"]["min"]
            max_calories=self.constraints["calories"]["max"]
        if("proteins" in self.constraints.keys()):
            min_proteins=self.constraints["proteins"]["min"]
            max_proteins=self.constraints["proteins"]["max"]
        if("fats" in self.constraints.keys()):
            min_fats=self.constraints["fats"]["min"]
            max_fats=self.constraints["fats"]["max"]
        
        
        penalty = 0
        if total_calories < min_calories or total_calories > max_calories:
            penalty += (abs(total_calories - min_calories) + abs(total_calories - max_calories))/abs(max_calories-min_calories)
        if total_proteins < min_proteins or total_proteins > max_proteins:
            penalty += (abs(total_proteins - min_proteins) + abs(total_proteins - max_proteins))/abs(max_proteins-min_proteins)
        if total_fats < min_fats or total_fats > max_fats:
            penalty += (abs(total_fats - min_fats) + abs(total_fats - max_fats))/abs(max_fats-min_fats)
    
        return penalty

    def fit(self,):
        x0 = self.ratios  
        result=minimize(self.objective, x0, method='SLSQP', constraints={'type': 'eq', 'fun': self.constraint_func}, bounds=[(0, None)] * len(x0))

        quantities=[round(i,2) for i in result.x]

        macronutrient_profile={
            "calories":round(np.dot(result.x, self.calories_per_gram),2),
            "proteins":round(np.dot(result.x, self.proteins_per_gram),2),
            "fats":round(np.dot(result.x, self.fats_per_gram),2)
        }
        return quantities,macronutrient_profile

