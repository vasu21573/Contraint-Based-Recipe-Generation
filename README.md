
# Recipe Generator with Macronutrient Constraints

This project builds on the groundwork established by the Ratatouille research initiative. Our work extends its scope by addressing a notable limitation: ensuring generated recipes meet specific nutritional goals. While the original Ratatouille system excels at generating realistic and novel recipes, our contribution focuses on adapting these recipes to comply with macronutrient constraints such as calories, proteins, fats, and more, further pushing the boundaries of state-of-the-art culinary AI.

---

## Architecture Overview

 Traditional large language models (LLMs), such as GPT-2, struggle at regression tasks like precise ingredient estimation. To overcome this, we adopt a hybrid approach that combines LLM-generated outputs with convex optimization techniques. Below is a detailed breakdown of the workflow:

   1. **LLM-Based Estimation**: The pretrained **Ratatouille GPT-2 model** is employed to:
      - Generate a recipe title and cooking instructions based on the provided list of ingredients and cuisine type.
      - Provide rough ingredient quantity estimates by analyzing ingredient relationships in recipes from its training data relevant to the cuisine specified.

   2. **Ratio-Based Initialization**:
      - The initial ingredient quantities are normalized into **ratios**, mimicking how chefs intuitively estimate proportions between ingredients.  
      - For example, the model might estimate a ratio of `1:2` for "flour" and "water" in a dough recipe, where water is twice as much as flour.

   3. **Convex Optimization Layer**: To refine these estimates, a convex optimization problem is defined, where:
      - Each ingredient is represented as a vector containing its macronutrient values (calories, proteins, and fats).
      - A multi-dimensional constraint function ensures that the total recipe adheres to the user-specified macronutrient constraints.
      - The objective function minimizes deviations from the constraints while preserving the initial ratios provided by the LLM.

   Optimization is implemented using `Sequential Least Squares Programming` technique.


---

## Directory Structure

- **GPT2_NEW/**
  - **`added_tokens.json`**
  - **`config.json`**
  - **`merges.txt`**
  - **`pytorch_model.bin`**
  - **`special_tokens_map.json`**
  - **`tokenizer.json`**
  - **`tokenizer_config.json`**
  - **`vocab.json`**
  - **`training_args.bin`**
- **resources/**
  - **`dataset_nutrients.json`**
  - [**`prompt_phrase2gram.txt`**](./resources/prompt_phrase2gram.txt): *Persona for the recipe generation process*
  - **`sample_ratatouille_output.txt`**
- [**`conversational_agent.py`**](conversational_agent.py): *Determines ingredient quantity ratios*
- [**`convex_optimisation.py`**](convex_optimisation.py): *Optimizes ingredient quantities*
- [**`main.ipynb`**](main.ipynb): *Demonstrates the functionality of the pipeline*
- [**`ratatouille_model_parser.py`**](ratatouille_model_parser.py): *Generates recipes using the pretrained model*
- **`requirements.txt`**


## Getting Started

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Set up your environment**:
   - Obtain an API key from [Hugging Face](https://huggingface.co/). This API key is essential for authenticating access to open source LLaMA models, which are utilized in this project. Once you have the key, set it in your environment:
     ```bash
     export HUGGINGFACE_API_KEY=<YOUR_TOKEN_HERE>
     ```

3. **Model Requirement**:
   - The project utilizes a pre-trained Ratatouille model for recipe generation, with its associated files located in the [`GPT2_NEW/`](GPT2_NEW/) directory.
   - If you replace the model in the future, ensure that the new model is trained with the necessary Ratatouille-specific tokenization tags to maintain compatibility with the pipeline. An example raw model output is provided for reference in [`here`](./resources/sample_ratatouille_ouptut.txt).

4. **Input Format**:
   - Ingredients:  
   The `INGREDIENTS` input must be provided as a **list of strings**. 
   Example:  
      ```python
      INGREDIENTS = ["milk", "sugar", "honey", "bread"]
      ```
   - Constraints:  
   The `CONSTRAINTS` input should be a **dictionary** specifying macronutrient ranges. You have the flexibility to define one or more constraints out of `calories`(kcal), `proteins`(g), `fats`(g). Example:  
      ```python
      CONSTRAINTS = {
         "calories": {"min": 200, "max": 300},
         "proteins": {"min": 20, "max": 50},
         "fats": {"min": 10, "max": 30},
      }
      ```
      - You can omit any macronutrient not relevant to your recipe, and it will be treated as unconstrained.  
      - At least one constraint is required to guide the recipe generation.

---

### About the Creators

This project, **Recipe Generator with Macronutrient Constraints**, was developed as part of an academic initiative under the [**CoSy Lab**](https://cosylab.iiitd.edu.in/) at **IIIT-Delhi**, during the Monsoon Semester of 2024.

**Contributors:**

- **[Vinayak](mailto:vinayak21574@iiitd.ac.in)** | [LinkedIn](https://www.linkedin.com/in/kayaniv)
- **[Vasu](mailto:vasu21573@iiitd.ac.in)**
- **[Mohit](mailto:mohit21542@iiitd.ac.in)**
- **[Parth](mailto:parth21548@iiitd.ac.in)**

**Citations:**
Mansi Goel, Pallab Chakraborty, Vijay Ponnaganti, Minnet Khan, Sritanaya Tatipamala, Aakanksha Saini, and Ganesh Bagler. Ratatouille: A tool for Novel Recipe Generation. IEEE 38th International Conference on Data Engineering Workshop (ICDEW) 2022 (https://ieeexplore.ieee.org/document/9814641).

The creators welcome questions, suggestions, and collaborations to improve or expand the scope of the project.
