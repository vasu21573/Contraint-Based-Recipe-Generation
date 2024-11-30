import os
import json
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Tuple


class Optimisation:
    def __init__(
        self,
        ingredients_name: List[str],
        ingredients_quantity: List[float],
        constraints: Dict[str, Dict[str, float]]
    ):
        """
        Initializes the Optimisation class.

        Args:
            ingredients_name (List[str]): List of ingredient names.
            ingredients_quantity (List[float]): Initial quantities of ingredients.
            constraints (Dict[str, Dict[str, float]]): Constraints for macronutrients.
        """
        self.constraints = constraints
        self.ratios = [q / ingredients_quantity[0] for q in ingredients_quantity]
        
        # Load nutrient data
        try:
            with open("dataset_nutrients.json") as f:
                macro_nutrients = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("Nutrient dataset file 'dataset_nutrients.json' not found.")
        except json.JSONDecodeError:
            raise ValueError("Error decoding 'dataset_nutrients.json'. Ensure it is a valid JSON file.")
        
        # Extract macronutrient values
        self.calories_per_gram = []
        self.proteins_per_gram = []
        self.fats_per_gram = []

        for ingredient in ingredients_name:
            nutrient_data = macro_nutrients.get(ingredient, {})
            self.calories_per_gram.append(nutrient_data.get("calories", 0))
            self.proteins_per_gram.append(nutrient_data.get("protein", 0))
            self.fats_per_gram.append(nutrient_data.get("fats", 0))

        # Convert lists to numpy arrays for efficient computation
        self.calories_per_gram = np.array(self.calories_per_gram)
        self.proteins_per_gram = np.array(self.proteins_per_gram)
        self.fats_per_gram = np.array(self.fats_per_gram)

    def constraint_func(self, x: np.ndarray) -> List[float]:
        """
        Constraint function to maintain ingredient ratios.

        Args:
            x (np.ndarray): Array of ingredient quantities.

        Returns:
            List[float]: List of constraint values for optimization.
        """
        base = x[0] / self.ratios[0]
        return [(x[i] - self.ratios[i] * base) for i in range(1, len(x))]

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function to minimize macronutrient constraint violations.

        Args:
            x (np.ndarray): Array of ingredient quantities.

        Returns:
            float: Calculated penalty value.
        """
        total_calories = np.dot(x, self.calories_per_gram)
        total_proteins = np.dot(x, self.proteins_per_gram)
        total_fats = np.dot(x, self.fats_per_gram)

        # Set default min/max values for macronutrients
        default_bounds = {"min": -1, "max": 1e8}
        calorie_bounds = self.constraints.get("calories", default_bounds)
        protein_bounds = self.constraints.get("proteins", default_bounds)
        fat_bounds = self.constraints.get("fats", default_bounds)

        # Calculate penalties
        penalty = 0
        if not calorie_bounds["min"] <= total_calories <= calorie_bounds["max"]:
            penalty += self._penalty(total_calories, calorie_bounds)
        if not protein_bounds["min"] <= total_proteins <= protein_bounds["max"]:
            penalty += self._penalty(total_proteins, protein_bounds)
        if not fat_bounds["min"] <= total_fats <= fat_bounds["max"]:
            penalty += self._penalty(total_fats, fat_bounds)

        return penalty

    @staticmethod
    def _penalty(value: float, bounds: Dict[str, float]) -> float:
        """
        Calculates the penalty for a value outside a given range.

        Args:
            value (float): Value to check.
            bounds (Dict[str, float]): Bounds with 'min' and 'max' keys.

        Returns:
            float: Calculated penalty.
        """
        return (
            max(0, bounds["min"] - value) + max(0, value - bounds["max"])
        ) / max(1, abs(bounds["max"] - bounds["min"]))

    def fit(self) -> Tuple[List[float], Dict[str, float]]:
        """
        Fits the optimization model to determine ingredient quantities.

        Returns:
            Tuple[List[float], Dict[str, float]]:
                - List of optimized ingredient quantities.
                - Macronutrient profile (calories, proteins, fats).
        """
        x0 = self.ratios  # Initial guess based on ratios
        bounds = [(0, None)] * len(x0)

        result = minimize(
            self.objective,
            x0,
            method="SLSQP",
            constraints={"type": "eq", "fun": self.constraint_func},
            bounds=bounds
        )

        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        optimized_quantities = [round(q, 2) for q in result.x]
        macronutrient_profile = {
            "calories": round(np.dot(result.x, self.calories_per_gram), 2),
            "proteins": round(np.dot(result.x, self.proteins_per_gram), 2),
            "fats": round(np.dot(result.x, self.fats_per_gram), 2),
        }

        return optimized_quantities, macronutrient_profile
