import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# Suppress specific warnings for cleaner output in Streamlit
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Streamlit App Configuration ---
st.set_page_config(
    page_title="ML Bias-Variance Explorer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Project Information in Sidebar First ---
st.sidebar.markdown("# 🧠 ML Bias-Variance Explorer")
st.sidebar.markdown("---")

with st.sidebar.expander("📋 Project Information", expanded=True):
    st.markdown("""
    **🎯 About This Project**
    
    An interactive educational tool to understand the fundamental bias-variance tradeoff in machine learning through hands-on experimentation.
    
    **✨ Features:**
    - Real-time bias-variance decomposition
    - Multiple ML algorithms comparison
    - VC Dimension visualization
    - Interactive parameter tuning
    
    **🛠️ Tech Stack:**
    - Python & Streamlit
    - Scikit-learn
    - Matplotlib & Seaborn
    
    **👨‍💻 Perfect For:**
    - ML Students & Educators
    - Data Science Learning
    - Interview Preparation
    - Research & Development
    
    **🚀 Created by:** ML Education Initiative
    
    **📖 Usage:** Adjust parameters below to explore real-time changes in model behavior!
    """)

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Simulation Controls")

# Set up Matplotlib and Seaborn for better aesthetics
sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

# --- Enhanced Data Generation Functions ---
def true_function_sin(x):
    return np.sin(2 * np.pi * x) * 2.5 + x * 0.5

def true_function_quadratic(x):
    return 2 * x**2 - 1.5 * x + 0.5

def true_function_step(x):
    return np.where(x < 0, 1.0, -1.0) + np.where(x > 0.5, 0.5, 0)

def true_function_complex_oscillation(x):
    return np.sin(4 * np.pi * x) * np.cos(2 * np.pi * x) * 3 + x

true_functions_map = {
    "🌊 Sine Wave": true_function_sin,
    "📈 Quadratic Curve": true_function_quadratic,
    "🔄 Step Function": true_function_step,
    "🌀 Complex Oscillation": true_function_complex_oscillation,
}

def generate_dataset(n_samples, selected_true_function, noise_std, x_range, is_test_set=False, seed=None):
    """Generate dataset with optional noise for bias-variance analysis."""
    if seed is not None:
        np.random.seed(seed)
        
    X = np.random.rand(n_samples) * (x_range[1] - x_range[0]) + x_range[0]
    y_true = selected_true_function(X)
    if is_test_set:
        y_noisy = y_true
    else:
        y_noisy = y_true + np.random.randn(n_samples) * noise_std
    return X.reshape(-1, 1), y_noisy, y_true

# Fixed parameter for test data density
_N_SAMPLES_TEST_DENSE = 300  # Reduced for better performance

def get_models_config(n_samples_train_for_knn):
    """Get model configurations with complexity parameters."""
    return {
        "Polynomial Regression": {
            "description": "Uses polynomial features to fit curves. Higher degree = more flexible curves.",
            "vc_proxy_info": "VC dimension ≈ degree + 1. Higher degree = higher capacity.",
            "complexity_param": "degree",
            "complexity_values": range(1, 12),  # Reduced range for performance
            "model_builder": lambda degree: make_pipeline(PolynomialFeatures(degree), LinearRegression()),
            "x_label": "Polynomial Degree"
        },
        "Decision Tree": {
            "description": "Splits data recursively. Max depth controls tree complexity.",
            "vc_proxy_info": "Deeper trees have higher capacity but risk overfitting.",
            "complexity_param": "max_depth",
            "complexity_values": range(1, 15),  # Reduced range
            "model_builder": lambda depth: DecisionTreeRegressor(max_depth=depth, random_state=42),
            "x_label": "Max Tree Depth"
        },
        "K-Nearest Neighbors": {
            "description": "Predicts using k nearest training points. Smaller k = more complex.",
            "vc_proxy_info": "1-NN memorizes data (infinite VC). Higher k = simpler model.",
            "complexity_param": "n_neighbors",
            "complexity_values": list(range(1, max(1, min(20, n_samples_train_for_knn))))[::-1],
            "model_builder": lambda n_neighbors: KNeighborsRegressor(n_neighbors=n_neighbors),
            "x_label": "Number of Neighbors (k)"
        },
        "Ridge Regression": {
            "description": "Linear regression with L2 regularization. Smaller alpha = less regularization.",
            "vc_proxy_info": "Lower alpha = higher effective capacity.",
            "complexity_param": "alpha",
            "complexity_values": [100, 10, 1, 0.1, 0.01, 0.001],  # Simplified range
            "model_builder": lambda alpha: Ridge(alpha=alpha, random_state=42),
            "x_label": "Ridge Alpha (regularization)"
        },
        "Gradient Boosting": {
            "description": "Ensemble of weak learners. More estimators = higher complexity.",
            "vc_proxy_info": "More trees = higher capacity and potential overfitting.",
            "complexity_param": "n_estimators",
            "complexity_values": range(10, 101, 15),  # Reduced range for performance
            "model_builder": lambda n_estimators: GradientBoostingRegressor(n_estimators=n_estimators, max_depth=3, random_state=42),
            "x_label": "Number of Estimators"
        }
    }

# --- VC Dimension Visualization Functions ---
@st.cache_data(show_spinner=False, ttl=3600)
def generate_vc_dimension_data(selected_model_key, _selected_true_function, n_samples_train, noise_std, x_range, random_seed):
    """Generate data for VC dimension visualization and analysis."""
    np.random.seed(random_seed)
    
    dynamic_models_config = get_models_config(n_samples_train)
    config = dynamic_models_config[selected_model_key]
    complexity_values = config["complexity_values"]
    model_builder = config["model_builder"]
    
    # Generate datasets of varying sizes
    dataset_sizes = [5, 10, 20, 50, 100, 200]
    vc_results = {
        'dataset_sizes': dataset_sizes,
        'perfect_fit_rates': [],
        'effective_vc_estimates': [],
        'generalization_gaps': []
    }
    
    for size in dataset_sizes:
        perfect_fits = 0
        gen_gaps = []
        
        # Test multiple random datasets
        for trial in range(20):  # Reduced for performance
            X_train, y_train, _ = generate_dataset(
                size, _selected_true_function, noise_std, x_range, 
                is_test_set=False, seed=random_seed + trial + size*100
            )
            X_test, y_test, _ = generate_dataset(
                min(50, size*2), _selected_true_function, noise_std, x_range,
                is_test_set=False, seed=random_seed + trial + size*100 + 1000
            )
            
            # Find most complex model that can fit this data
            for complexity in reversed(complexity_values):
                try:
                    model = model_builder(complexity)
                    model.fit(X_train, y_train)
                    
                    train_pred = model.predict(X_train)
                    train_mse = mean_squared_error(y_train, train_pred)
                    
                    test_pred = model.predict(X_test)
                    test_mse = mean_squared_error(y_test, test_pred)
                    
                    # Check if model achieves perfect fit (low training error)
                    if train_mse < 0.01:  # Threshold for "perfect" fit
                        perfect_fits += 1
                        gen_gaps.append(test_mse - train_mse)
                        break
                        
                except Exception:
                    continue
        
        # Calculate metrics for this dataset size
        perfect_fit_rate = perfect_fits / 20.0
        vc_results['perfect_fit_rates'].append(perfect_fit_rate)
        
        # Estimate effective VC dimension (size where perfect fit rate drops below 0.5)
        effective_vc = size if perfect_fit_rate >= 0.5 else 0
        vc_results['effective_vc_estimates'].append(effective_vc)
        
        # Average generalization gap
        avg_gap = np.mean(gen_gaps) if gen_gaps else np.nan
        vc_results['generalization_gaps'].append(avg_gap)
    
    return vc_results

# --- Optimized Simulation Functions ---
@st.cache_data(show_spinner=False, ttl=3600)  # Added TTL for memory management
def run_bias_variance_simulation(selected_model_key, _selected_true_function, n_training_sets, n_samples_train, noise_std, x_range, random_seed):
    """Run bias-variance simulation for a single model."""
    np.random.seed(random_seed)
    
    dynamic_models_config = get_models_config(n_samples_train)
    
    X_test_dense, y_test_true_for_bias, y_test_true_actual = generate_dataset(
        _N_SAMPLES_TEST_DENSE, _selected_true_function, noise_std, x_range, is_test_set=True, seed=random_seed
    )
    _, y_test_noisy_for_total_error, _ = generate_dataset(
        _N_SAMPLES_TEST_DENSE, _selected_true_function, noise_std, x_range, is_test_set=False, seed=random_seed + 1 
    )

    sort_idx = np.argsort(X_test_dense.flatten())
    X_test_sorted = X_test_dense[sort_idx]
    y_test_true_actual_sorted = y_test_true_actual[sort_idx]
    y_test_noisy_for_total_error_sorted = y_test_noisy_for_total_error[sort_idx]

    irreducible_error = noise_std**2
    config_dict = dynamic_models_config[selected_model_key]
    complexity_values = config_dict["complexity_values"]
    model_builder = config_dict["model_builder"]

    if not complexity_values:
        return None 

    train_errors = []
    test_errors = []
    biases_sq = []
    variances = []
    
    complexity_indices_for_viz = [0]
    if len(complexity_values) > 1:
        complexity_indices_for_viz.append(len(complexity_values) // 2)
    if len(complexity_values) > 2:
        complexity_indices_for_viz.append(len(complexity_values) - 1)

    viz_complexity_levels = [complexity_values[idx] for idx in complexity_indices_for_viz]
    all_test_predictions = {lvl: {'avg_pred': None, 'individual_preds': []} for lvl in viz_complexity_levels}

    for i, complexity_value in enumerate(complexity_values):
        predictions_across_sets = []
        current_train_errors = []
        training_set_seeds = np.arange(n_training_sets) + random_seed * 1000 + i * n_training_sets

        for j in range(n_training_sets):
            X_train, y_train, _ = generate_dataset(n_samples_train, _selected_true_function, noise_std, x_range, is_test_set=False, seed=int(training_set_seeds[j]))
            model = model_builder(complexity_value)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            current_train_errors.append(mean_squared_error(y_train, y_train_pred))

            y_test_pred = model.predict(X_test_sorted)
            predictions_across_sets.append(y_test_pred)
        
        train_errors.append(np.mean(current_train_errors))
        predictions_across_sets = np.array(predictions_across_sets)
        avg_pred_on_test = np.mean(predictions_across_sets, axis=0)
        
        if complexity_value in viz_complexity_levels:
            all_test_predictions[complexity_value]['avg_pred'] = avg_pred_on_test
            num_individual_preds_to_show = min(8, n_training_sets)  # Reduced for performance
            all_test_predictions[complexity_value]['individual_preds'] = \
                predictions_across_sets[np.random.choice(n_training_sets, num_individual_preds_to_show, replace=False)]

        bias_sq = np.mean((avg_pred_on_test - y_test_true_for_bias)**2)
        biases_sq.append(bias_sq)

        variance = np.mean(np.var(predictions_across_sets, axis=0))
        variances.append(variance)
        
        total_test_error = bias_sq + variance + irreducible_error
        test_errors.append(total_test_error)

    return {
        "train_errors": train_errors,
        "test_errors": test_errors,
        "biases_sq": biases_sq,
        "variances": variances,
        "all_test_predictions": all_test_predictions,
        "irreducible_error": irreducible_error,
        "X_test_sorted": X_test_sorted,
        "y_test_true_actual_sorted": y_test_true_actual_sorted,
        "y_test_noisy_for_total_error_sorted": y_test_noisy_for_total_error_sorted,
        "y_test_true_for_bias": y_test_true_for_bias, 
        "X_range_viz": x_range 
    }

@st.cache_data(show_spinner=False, ttl=3600)
def run_training_size_effect_simulation(selected_model_key, _selected_true_function, fixed_complexity_value, n_training_sets, max_n_samples_train, noise_std, x_range, random_seed):
    """Simulate effect of training set size on model performance."""
    np.random.seed(random_seed)
    
    dynamic_models_config = get_models_config(max_n_samples_train) 
    model_builder = dynamic_models_config[selected_model_key]["model_builder"]
    
    X_test_dense, y_test_true_for_bias, _ = generate_dataset(
        _N_SAMPLES_TEST_DENSE, _selected_true_function, noise_std, x_range, is_test_set=True, seed=random_seed + 300
    )
    sort_idx = np.argsort(X_test_dense.flatten())
    X_test_sorted = X_test_dense[sort_idx]
    y_test_true_for_bias_sorted = y_test_true_for_bias[sort_idx]

    irreducible_error = noise_std**2
    n_samples_train_values = list(range(5, max_n_samples_train + 1, max(1, max_n_samples_train // 8)))  # Reduced points
    
    if not n_samples_train_values or n_samples_train_values[0] > max_n_samples_train:
         n_samples_train_values = [max_n_samples_train] if max_n_samples_train >= 5 else [5] 
    
    if len(n_samples_train_values) == 0:
        return None

    test_errors = []
    biases_sq = []
    variances = []

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, current_n_samples_train in enumerate(n_samples_train_values):
        status_text.text(f"Training set size simulation: {current_n_samples_train} samples ({i+1}/{len(n_samples_train_values)})")
        progress_bar.progress((i + 1) / len(n_samples_train_values))
        
        predictions_across_sets = []
        training_set_seeds = np.arange(n_training_sets) + random_seed * 2000 + i * n_training_sets

        for j in range(n_training_sets):
            X_train, y_train, _ = generate_dataset(current_n_samples_train, _selected_true_function, noise_std, x_range, is_test_set=False, seed=int(training_set_seeds[j]))
            
            if selected_model_key == "🎯 K-Nearest Neighbors":
                effective_k = fixed_complexity_value
                if fixed_complexity_value >= current_n_samples_train:
                    effective_k = max(1, current_n_samples_train - 1)
                if effective_k == 0: continue
                model = model_builder(effective_k)
            else:
                model = model_builder(fixed_complexity_value)

            if X_train.shape[0] > 0:
                try:
                    model.fit(X_train, y_train)
                    y_test_pred = model.predict(X_test_sorted)
                    predictions_across_sets.append(y_test_pred)
                except Exception:
                    pass
        
        if not predictions_across_sets:
            test_errors.append(np.nan)
            biases_sq.append(np.nan)
            variances.append(np.nan)
            continue

        predictions_across_sets = np.array(predictions_across_sets)
        avg_pred_on_test = np.mean(predictions_across_sets, axis=0)
        
        bias_sq = np.mean((avg_pred_on_test - y_test_true_for_bias_sorted)**2)
        biases_sq.append(bias_sq)

        variance = np.mean(np.var(predictions_across_sets, axis=0))
        variances.append(variance)
        
        total_test_error = bias_sq + variance + irreducible_error
        test_errors.append(total_test_error)
    
    progress_bar.empty()
    status_text.empty()

    return {
        "n_samples_train_values": n_samples_train_values,
        "test_errors": test_errors,
        "biases_sq": biases_sq,
        "variances": variances,
        "irreducible_error": irreducible_error
    }

def plot_residuals_analysis(model, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Comprehensive residual analysis for model diagnostics.
    
    Parameters:
    -----------
    model : fitted sklearn model
        The trained model to analyze
    X_train, y_train : array-like
        Training data
    X_test, y_test : array-like  
        Test data
    model_name : str
        Name of the model for plot titles
    
    Returns:
    --------
    fig : matplotlib figure
        The residual analysis plot
    """
    from scipy import stats
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Get predictions
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    # Calculate residuals
    train_residuals = y_train - train_pred
    test_residuals = y_test - test_pred
    
    # Training residuals vs predictions
    axes[0,0].scatter(train_pred, train_residuals, alpha=0.6, color='blue')
    axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0,0].set_xlabel('Predicted Values')
    axes[0,0].set_ylabel('Residuals')
    axes[0,0].set_title(f'Training Residuals vs Predictions ({model_name})')
    axes[0,0].grid(True, alpha=0.3)
    
    # Test residuals vs predictions  
    axes[0,1].scatter(test_pred, test_residuals, alpha=0.6, color='orange')
    axes[0,1].axhline(y=0, color='red', linestyle='--', alpha=0.8)
    axes[0,1].set_xlabel('Predicted Values')
    axes[0,1].set_ylabel('Residuals')
    axes[0,1].set_title(f'Test Residuals vs Predictions ({model_name})')
    axes[0,1].grid(True, alpha=0.3)
    
    # Residual distributions
    axes[1,0].hist(train_residuals, bins=20, alpha=0.7, label='Train', color='blue', density=True)
    axes[1,0].hist(test_residuals, bins=20, alpha=0.7, label='Test', color='orange', density=True)
    axes[1,0].set_xlabel('Residuals')
    axes[1,0].set_ylabel('Density')
    axes[1,0].set_title(f'Residual Distributions ({model_name})')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Q-Q plot for normality check
    stats.probplot(test_residuals, dist="norm", plot=axes[1,1])
    axes[1,1].set_title(f'Q-Q Plot - Test Residuals ({model_name})')
    axes[1,1].grid(True, alpha=0.3)
    
    # Add summary statistics
    train_rmse = np.sqrt(np.mean(train_residuals**2))
    test_rmse = np.sqrt(np.mean(test_residuals**2))
    
    fig.suptitle(f'Residual Analysis: {model_name}\nTrain RMSE: {train_rmse:.3f} | Test RMSE: {test_rmse:.3f}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


@st.cache_data(show_spinner=False, ttl=3600)
def run_all_models_bias_variance_simulations(_selected_true_function, n_training_sets, n_samples_train, noise_std, x_range, random_seed):
    """Run simulations for all models."""
    all_sim_results = {}
    dynamic_models_config = get_models_config(n_samples_train)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_key in enumerate(dynamic_models_config.keys()):
        status_text.text(f"Running simulation: {model_key}")
        progress_bar.progress((i + 1) / len(dynamic_models_config))
        
        result = run_bias_variance_simulation(
            model_key, _selected_true_function, n_training_sets, n_samples_train, noise_std, x_range, random_seed
        )
        if result:
            all_sim_results[model_key] = result
    
    progress_bar.empty()
    status_text.empty()
    return all_sim_results

# --- Enhanced UI Layout ---
st.title("🧠 Advanced Bias-Variance Tradeoff Explorer")

# Quick info banner
st.info("🎯 **Interactive ML Learning Tool** | Explore how model complexity affects bias vs variance through real-time simulations")

# Main parameter controls in a more compact layout
with st.expander("🎛️ Global Simulation Parameters", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        n_training_sets = st.slider(
            "Training Sets", 20, 100, 50, 10,
            help="Number of training sets for robust averaging"
        )
    with col2:
        n_samples_train = st.slider(
            "Samples per Set", 5, 40, 20, 1,
            help="Training samples per dataset"
        )
    with col3:
        noise_std = st.slider(
            "Noise Level", 0.1, 1.2, 0.6, 0.1,
            help="Data noise (irreducible error)"
        )
    with col4:
        random_seed = st.number_input(
            "Random Seed", 0, 999, 42, 1,
            help="For reproducible results"
        )

# Sidebar controls
selected_true_function_name = st.sidebar.selectbox(
    "🎯 True Function",
    list(true_functions_map.keys()),
    help="Select underlying data pattern"
)
selected_true_function = true_functions_map[selected_true_function_name]

x_range = st.sidebar.slider(
    "📊 X-axis Range",
    -2.0, 2.0, (-1.5, 1.5), 0.1,
    help="Input feature range"
)

# --- Enhanced Data Visualization ---
st.header("📈 The Learning Challenge")

col1, col2 = st.columns([2, 1])

with col1:
    X_example_train, y_example_train, _ = generate_dataset(
        n_samples_train * 2, selected_true_function, noise_std, x_range, is_test_set=False, seed=int(random_seed + 100)
    )
    X_test_initial_plot, y_test_initial_plot_true, _ = generate_dataset(
        _N_SAMPLES_TEST_DENSE, selected_true_function, 0, x_range, is_test_set=True, seed=int(random_seed + 200) 
    )
    sort_idx_initial = np.argsort(X_test_initial_plot.flatten())

    fig_data_viz, ax_data_viz = plt.subplots(figsize=(10, 5))
    ax_data_viz.plot(X_test_initial_plot[sort_idx_initial], y_test_initial_plot_true[sort_idx_initial], 
                     label='True Function', color='darkgreen', linewidth=3)
    ax_data_viz.scatter(X_example_train, y_example_train, 
                       label='Noisy Training Data', color='steelblue', alpha=0.6, s=40)
    ax_data_viz.set_title(f'Learning from Noisy Observations: {selected_true_function_name}')
    ax_data_viz.set_xlabel('Input Feature (X)')
    ax_data_viz.set_ylabel('Target Value (Y)')
    ax_data_viz.legend()
    ax_data_viz.grid(True, alpha=0.3)
    st.pyplot(fig_data_viz)
    plt.close(fig_data_viz)

with col2:
    st.metric("🎯 Irreducible Error", f"{noise_std**2:.3f}")
    st.metric("📊 Training Samples", n_samples_train)
    st.metric("🔄 Simulation Sets", n_training_sets)
    
    st.markdown("""
    **💡 Key Challenge:**
    Models must learn the hidden pattern from noisy data while avoiding overfitting to noise.
    """)

# --- Model Selection and Analysis ---
st.header("🤖 Model Analysis")

selected_model_key = st.selectbox(
    "Choose ML Algorithm",
    list(get_models_config(n_samples_train).keys()),
    help="Different algorithms show unique bias-variance characteristics"
)

current_model_config = get_models_config(n_samples_train)[selected_model_key]

# Enhanced model info display
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**{selected_model_key}:** {current_model_config['description']}")
        st.markdown(f"**Complexity Control:** {current_model_config['vc_proxy_info']}")
    with col2:
        if st.button("🚀 Run Simulation", type="primary"):
            st.session_state.run_simulation = True

# Run simulation with better feedback
# if st.session_state.get('run_simulation', False):
with st.spinner(f"🔄 Analyzing {selected_model_key}..."):
    results = run_bias_variance_simulation(
        selected_model_key, selected_true_function, n_training_sets, n_samples_train, noise_std, x_range, random_seed
    )

st.session_state.run_simulation = False

if results is None:
    st.error("❌ Insufficient data for analysis. Please adjust parameters.")
    st.stop()

# Enhanced results display
train_errors = results["train_errors"]
test_errors = results["test_errors"]
biases_sq = results["biases_sq"]
variances = results["variances"]
all_test_predictions = results["all_test_predictions"]
irreducible_error_sim = results["irreducible_error"] 
X_test_sorted = results["X_test_sorted"]
y_test_true_actual_sorted = results["y_test_true_actual_sorted"]

# Main bias-variance plot with enhanced styling
st.subheader("📊 Bias-Variance Tradeoff Analysis")

fig_tradeoff, ax_tradeoff = plt.subplots(figsize=(12, 6))

# Enhanced plot styling
ax_tradeoff.plot(current_model_config["complexity_values"], train_errors, 
                    label='Training Error', color='#2E86AB', linewidth=2.5, marker='s', markersize=4)
ax_tradeoff.plot(current_model_config["complexity_values"], test_errors, 
                    label='Test Error (Key!)', color='#A23B72', linewidth=3.5, marker='o', markersize=5)
ax_tradeoff.plot(current_model_config["complexity_values"], biases_sq, 
                    label='Bias²', color='#F18F01', linestyle='--', linewidth=2)
ax_tradeoff.plot(current_model_config["complexity_values"], variances, 
                    label='Variance', color='#C73E1D', linestyle='--', linewidth=2)
ax_tradeoff.axhline(y=irreducible_error_sim, color='gray', linestyle=':', 
                    label=f'Irreducible Error ({irreducible_error_sim:.3f})', linewidth=2)

optimal_idx = np.argmin(test_errors)
optimal_complexity = current_model_config["complexity_values"][optimal_idx]
ax_tradeoff.axvline(x=optimal_complexity, color='black', linestyle=':', 
                    label=f'Optimal: {optimal_complexity}', linewidth=2)

ax_tradeoff.set_xlabel(current_model_config["x_label"], fontsize=12)
ax_tradeoff.set_ylabel('Error (MSE)', fontsize=12)
ax_tradeoff.set_title(f'Bias-Variance Analysis: {selected_model_key}', fontsize=14, fontweight='bold')
ax_tradeoff.legend(loc='upper right', framealpha=0.9)
ax_tradeoff.grid(True, alpha=0.3)

# Add annotations for key regions
complexity_range = current_model_config["complexity_values"]
y_max = ax_tradeoff.get_ylim()[1]

# Underfitting region
ax_tradeoff.annotate('Underfitting\n(High Bias)', 
                    xy=(complexity_range[len(complexity_range)//4], y_max * 0.85),
                    ha='center', va='top', fontsize=10, color='darkred',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

# Overfitting region
ax_tradeoff.annotate('Overfitting\n(High Variance)', 
                    xy=(complexity_range[len(complexity_range)*3//4], y_max * 0.85),
                    ha='center', va='top', fontsize=10, color='darkred',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))

st.pyplot(fig_tradeoff)
plt.close(fig_tradeoff)

# Key insights in metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🎯 Optimal Complexity", str(optimal_complexity))
with col2:
    st.metric("📉 Min Test Error", f"{min(test_errors):.3f}")
with col3:
    st.metric("📊 Final Bias²", f"{biases_sq[-1]:.3f}")
with col4:
    st.metric("📈 Final Variance", f"{variances[-1]:.3f}")

# Visual model fits comparison
st.subheader("🎭 Model Behavior Visualization")

fig_fits, axes_fits = plt.subplots(1, 3, figsize=(15, 5))

# Use the same true function that was used in the simulation
X_train_example, y_train_example, _ = generate_dataset(
    n_samples_train, selected_true_function, noise_std, x_range, is_test_set=False, seed=int(random_seed * 2000)
)

viz_complexity_levels = list(all_test_predictions.keys())

behavior_labels = {
    viz_complexity_levels[0]: "Underfitting",
    viz_complexity_levels[1] if len(viz_complexity_levels) > 1 else viz_complexity_levels[0]: "Balanced",
    viz_complexity_levels[2] if len(viz_complexity_levels) > 2 else viz_complexity_levels[-1]: "Overfitting"
}

for j, comp_val in enumerate(viz_complexity_levels[:3]):
    ax = axes_fits[j]
    
    avg_pred = all_test_predictions[comp_val]['avg_pred']
    individual_preds = all_test_predictions[comp_val]['individual_preds']

    # Use y_test_true_actual_sorted from results which contains the correct true function values
    ax.plot(X_test_sorted, y_test_true_actual_sorted, 
            label='True Function', color='darkgreen', linewidth=3, linestyle='--')
    ax.scatter(X_train_example, y_train_example, 
                label='Training Data', color='steelblue', alpha=0.7, s=50)
    
    for ind_pred in individual_preds:
        ax.plot(X_test_sorted, ind_pred, color='orange', alpha=0.15, linewidth=1.5) 
    
    ax.plot(X_test_sorted, avg_pred, 
            label='Average Model', color='crimson', linewidth=4)
    
    ax.set_title(f'{behavior_labels.get(comp_val, f"Complexity: {comp_val}")}', 
                fontsize=12, fontweight='bold')
    ax.set_xlabel('Input (X)')
    ax.set_ylabel('Output (Y)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_fits)
plt.close(fig_fits)

# Enhanced insights
with st.expander("🧠 Key Insights from Analysis", expanded=True):
    st.markdown(f"""
    **📊 Performance Summary for {selected_model_key}:**
    
    - **🎯 Optimal Complexity:** {optimal_complexity} ({current_model_config['complexity_param']})
    - **📉 Best Test Error:** {min(test_errors):.3f} (vs Irreducible: {irreducible_error_sim:.3f})
    - **🔄 Error Improvement:** {((test_errors[0] - min(test_errors))/test_errors[0]*100):.1f}% better than simplest model
    
    **🎭 Behavior Patterns:**
    - **Left Plot (Underfitting):** Model too simple → misses true pattern → high bias
    - **Center Plot (Balanced):** Good complexity → captures pattern without noise → optimal generalization  
    - **Right Plot (Overfitting):** Model too complex → memorizes noise → high variance
    
    **💡 Practical Takeaway:** The optimal model balances bias and variance to minimize test error!
    """)

if results is not None:
    st.subheader("🔍 Residual Analysis & Model Diagnostics")
    
    with st.expander("📊 Understanding Residual Plots"):
        st.markdown("""
        **Residual analysis helps identify:**
        - **Heteroscedasticity**: Non-constant variance (funnel patterns)
        - **Non-linearity**: Curved patterns in residuals
        - **Outliers**: Points far from the residual center line
        - **Normality**: Q-Q plot should follow diagonal line
        
        **Good models show:**
        - Random scatter around zero
        - Constant variance across prediction range
        - Normally distributed residuals
        """)
    
    # Demonstrate residual analysis with optimal model
    optimal_idx = np.argmin(results["test_errors"])
    optimal_complexity = current_model_config["complexity_values"][optimal_idx]
    
    # Generate fresh data for residual analysis
    X_train_residual, y_train_residual, _ = generate_dataset(
        n_samples_train, selected_true_function, noise_std, x_range, 
        is_test_set=False, seed=random_seed + 500
    )
    X_test_residual, y_test_residual, _ = generate_dataset(
        50, selected_true_function, noise_std, x_range, 
        is_test_set=False, seed=random_seed + 600
    )
    
    # Fit optimal model
    optimal_model = current_model_config["model_builder"](optimal_complexity)
    optimal_model.fit(X_train_residual, y_train_residual)
    
    # Create residual plots
    fig_residuals = plot_residuals_analysis(
        optimal_model, X_train_residual, y_train_residual, 
        X_test_residual, y_test_residual, 
        f"{selected_model_key} (Optimal)"
    )
    
    st.pyplot(fig_residuals)
    plt.close(fig_residuals)
    
    # Compare with overfitted model
    if len(current_model_config["complexity_values"]) > 1:
        st.subheader("🔄 Overfitting vs Optimal Model Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Overfitted model (highest complexity)
            high_complexity = current_model_config["complexity_values"][-1]
            overfitted_model = current_model_config["model_builder"](high_complexity)
            overfitted_model.fit(X_train_residual, y_train_residual)
            
            fig_overfitted = plot_residuals_analysis(
                overfitted_model, X_train_residual, y_train_residual,
                X_test_residual, y_test_residual,
                f"{selected_model_key} (Overfitted)"
            )
            st.pyplot(fig_overfitted)
            plt.close(fig_overfitted)
        
        with col2:
            # Simple model (lowest complexity)
            low_complexity = current_model_config["complexity_values"][0]
            simple_model = current_model_config["model_builder"](low_complexity)
            simple_model.fit(X_train_residual, y_train_residual)
            
            fig_simple = plot_residuals_analysis(
                simple_model, X_train_residual, y_train_residual,
                X_test_residual, y_test_residual,
                f"{selected_model_key} (Underfitted)"
            )
            st.pyplot(fig_simple)
            plt.close(fig_simple)
        
        st.markdown("""
        **💡 Comparison Insights:**
        - **Underfitted models**: Large, systematic residual patterns
        - **Optimal models**: Random residual scatter, good normality
        - **Overfitted models**: Better training residuals, worse test patterns
        """)
        
    # VC Dimension Analysis Section
    st.subheader("🧮 VC Dimension Analysis")

    with st.expander("📊 Understanding VC Dimension", expanded=False):
        st.markdown("""
        **VC Dimension Practical Analysis:**
        
        This analysis estimates the effective VC dimension by testing the model's ability to achieve perfect fits on datasets of varying sizes.
        
        **Key Insights:**
        - **Shattering Ability**: Can the model perfectly fit random datasets?
        - **Data Requirements**: How much data is needed for reliable generalization?
        - **Capacity Control**: How does model complexity affect learning capacity?
        
        **Interpretation:**
        - High perfect fit rates on large datasets → High VC dimension
        - Large generalization gaps → Need more regularization
        - Rapid performance degradation → Model complexity mismatch
        """)

    if "run_vc_analysis" not in st.session_state:
        st.session_state.run_vc_analysis = False

    if "run_simulation" not in st.session_state:
        st.session_state.run_simulation = False

    if st.button("🔍 Analyze VC Dimension", help="Estimate effective VC dimension through shattering experiments"):
        st.session_state.run_vc_analysis = True

    if st.session_state.get('run_vc_analysis', False) and not st.session_state.get('run_simulation', False):
        with st.spinner("🧮 Running VC dimension analysis..."):
            vc_results = generate_vc_dimension_data(
                selected_model_key, 
                selected_true_function, 
                n_samples_train, 
                noise_std, 
                x_range, 
                random_seed
            )
        
        st.session_state.run_vc_analysis = False
        
        if vc_results:
            # Plot VC dimension analysis
            fig_vc, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Perfect fit rates
            ax1.plot(vc_results['dataset_sizes'], vc_results['perfect_fit_rates'], 
                    'bo-', linewidth=3, markersize=8, label='Perfect Fit Rate')
            ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                        label='50% Threshold')
            ax1.set_xlabel('Dataset Size')
            ax1.set_ylabel('Perfect Fit Rate')
            ax1.set_title(f'Model Capacity Analysis: {selected_model_key}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # Generalization gaps
            valid_gaps = [gap for gap in vc_results['generalization_gaps'] if not np.isnan(gap)]
            valid_sizes = [size for i, size in enumerate(vc_results['dataset_sizes']) 
                            if not np.isnan(vc_results['generalization_gaps'][i])]
            
            if valid_gaps:
                ax2.plot(valid_sizes, valid_gaps, 'ro-', linewidth=3, markersize=8, 
                        label='Generalization Gap')
                ax2.set_xlabel('Dataset Size')
                ax2.set_ylabel('Test Error - Training Error')
                ax2.set_title('Generalization Gap Analysis')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig_vc)
            plt.close(fig_vc)
            
            # Display key metrics
            # Estimate effective VC dimension (largest size with >50% perfect fit rate)
            effective_vc = 0
            for i, rate in enumerate(vc_results['perfect_fit_rates']):
                if rate >= 0.5:
                    effective_vc = vc_results['dataset_sizes'][i]
            
            st.metric("🎯 Estimated VC Dimension", 
                        f"~{effective_vc}" if effective_vc > 0 else "Unknown")
            st.metric("📊 Max Perfect Fit Rate", f"{max(vc_results['perfect_fit_rates']):.2f}")
            
            if valid_gaps:
                avg_gap = np.mean(valid_gaps)
                st.metric("⚖️ Avg Generalization Gap", f"{avg_gap:.3f}")
            
            # Insights based on results
            st.markdown("**🧠 VC Dimension Insights:**")
            
            if effective_vc == 0:
                st.warning("⚠️ Model cannot achieve perfect fits - may be undercapacity or heavily regularized")
            elif effective_vc < n_samples_train // 2:
                st.info(f"✅ Conservative capacity: Effective VC ({effective_vc}) << Training size ({n_samples_train})")
            elif effective_vc >= n_samples_train:
                st.error(f"🚨 High capacity risk: Effective VC ({effective_vc}) ≥ Training size ({n_samples_train})")
            else:
                st.success(f"🎯 Balanced capacity: Effective VC ({effective_vc}) appropriate for training size ({n_samples_train})")
            
            if valid_gaps and max(valid_gaps) > min(test_errors) * 2:
                st.warning("📈 Large generalization gaps detected - consider more regularization")
            
            st.markdown("""
            **🎓 VC Dimension Key Concepts:**
            
            **Shattering**: A model "shatters" a dataset if it can perfectly fit any possible labeling of those points.
            
            **Capacity Control**: Higher VC dimension means more model capacity but requires more data for reliable generalization.
            
            **Practical Rule**: For reliable learning, you typically need training data size ≫ VC dimension.
            
            **Model-Specific Notes:**
            - **Linear models**: VC ≈ # parameters
            - **Polynomial**: VC ≈ degree + 1  
            - **Trees**: Can have infinite VC dimension
            - **Neural nets**: VC grows with # parameters
            """)
    # st.success("✅ Residual analysis complete! Check plots for model diagnostics.")
    # st.balloons()


    
# Training set size analysis
with st.expander("📈 Training Set Size Effect Analysis"):
    st.markdown("Explore how more training data affects model performance at fixed complexity.")
    
    model_complexity_values = get_models_config(n_samples_train)[selected_model_key]["complexity_values"]
    
    if model_complexity_values:
        col1, col2 = st.columns(2)
        with col1:
            fixed_complexity = st.select_slider(
                f"Fixed {current_model_config['complexity_param']}:",
                options=list(model_complexity_values),
                value=model_complexity_values[len(model_complexity_values)//2]
            )
        with col2:
            max_samples = st.slider("Max Training Samples:", 30, 120, 80, 10)
        
        if st.button("🔍 Analyze Training Size Effect"):
            size_results = run_training_size_effect_simulation(
                selected_model_key, selected_true_function, fixed_complexity, 
                n_training_sets, max_samples, noise_std, x_range, random_seed
            )
            
            if size_results:
                fig_size, ax_size = plt.subplots(figsize=(10, 5))
                
                valid_indices = ~np.isnan(size_results["test_errors"])
                n_samples_valid = np.array(size_results["n_samples_train_values"])[valid_indices]
                test_errors_valid = np.array(size_results["test_errors"])[valid_indices]
                biases_sq_valid = np.array(size_results["biases_sq"])[valid_indices]
                variances_valid = np.array(size_results["variances"])[valid_indices]

                ax_size.plot(n_samples_valid, test_errors_valid, 'ro-', label='Test Error', linewidth=2.5)
                ax_size.plot(n_samples_valid, biases_sq_valid, 'g--', label='Bias²', linewidth=2)
                ax_size.plot(n_samples_valid, variances_valid, 'm--', label='Variance', linewidth=2)
                ax_size.axhline(y=size_results["irreducible_error"], color='gray', linestyle=':', 
                               label='Irreducible Error', linewidth=2)

                ax_size.set_xlabel('Training Set Size')
                ax_size.set_ylabel('Error (MSE)')
                ax_size.set_title(f'Impact of Training Data Size ({selected_model_key})')
                ax_size.legend()
                ax_size.grid(True, alpha=0.3)
                
                st.pyplot(fig_size)
                plt.close(fig_size)
                
                st.success("💡 **Key Insight:** More training data typically reduces variance while bias remains relatively constant!")

# Model comparison section
st.header("⚖️ Model Comparison Arena")

if st.button("🏆 Compare All Models", type="primary"):
    with st.spinner("🔄 Running comprehensive model comparison..."):
        all_sim_results = run_all_models_bias_variance_simulations(
            selected_true_function, n_training_sets, n_samples_train, noise_std, x_range, random_seed
        )

    if all_sim_results:
        st.success(f"✅ Compared {len(all_sim_results)} models successfully!")
        
        # Performance comparison table
        comparison_data = []
        for model_key, results_m in all_sim_results.items():
            if results_m:
                min_test_error = min(results_m["test_errors"])
                optimal_idx = np.argmin(results_m["test_errors"])
                config_m = get_models_config(n_samples_train)[model_key]
                optimal_complexity = config_m["complexity_values"][optimal_idx]
                
                comparison_data.append({
                    "Model": model_key,
                    "Best Test Error": f"{min_test_error:.3f}",
                    "Optimal Complexity": optimal_complexity,
                    "Final Bias²": f"{results_m['biases_sq'][-1]:.3f}",
                    "Final Variance": f"{results_m['variances'][-1]:.3f}"
                })
        
        st.dataframe(comparison_data, use_container_width=True)
        
        # Visual comparison
        num_models = len(all_sim_results)
        if num_models <= 4:
            n_rows, n_cols = 2, 2
        else:
            n_rows, n_cols = 2, 3
        
        fig_all, axes_all = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        axes_all = axes_all.flatten() if n_rows > 1 else [axes_all] if num_models == 1 else axes_all

        for i, (model_key, results_m) in enumerate(sorted(all_sim_results.items())):
            if i >= len(axes_all): break
            ax = axes_all[i]
            
            if results_m:
                config_m = get_models_config(n_samples_train)[model_key]
                ax.plot(config_m["complexity_values"], results_m["test_errors"], 'ro-', linewidth=2, markersize=3)
                ax.plot(config_m["complexity_values"], results_m["biases_sq"], 'g--', linewidth=1.5, alpha=0.7)
                ax.plot(config_m["complexity_values"], results_m["variances"], 'm--', linewidth=1.5, alpha=0.7)
                
                optimal_idx = np.nanargmin(results_m["test_errors"])
                optimal_complexity = config_m["complexity_values"][optimal_idx]
                ax.axvline(x=optimal_complexity, color='black', linestyle=':', alpha=0.8)
                
                ax.set_title(model_key.split()[-1], fontsize=10, fontweight='bold')  # Shorter titles
                ax.set_xlabel(config_m["complexity_param"], fontsize=8)
                ax.set_ylabel('Error', fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)

        # Hide unused subplots
        for i in range(num_models, len(axes_all)):
            fig_all.delaxes(axes_all[i])

        plt.tight_layout()
        st.pyplot(fig_all)
        plt.close(fig_all)
        
        st.success("🎯 **Insight:** Each model type has unique bias-variance characteristics. The best choice depends on your data and problem!")
    


# Enhanced sidebar footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠️ Quick Actions")
if st.sidebar.button("🗑️ Clear Cache"):
    st.cache_data.clear()
    st.sidebar.success("✅ Cache cleared!")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: #666;'>
<small>
🎓 Educational Tool<br>
⭐ Star on GitHub<br>
🤝 Contribute & Share
</small>
</div>
""", unsafe_allow_html=True)

# Footer with key takeaways
st.markdown("---")

def neural_network_playground():
    st.header("🧠 Neural Network Playground")
    st.write("Experiment with different neural network architectures and see how they perform on regression tasks.")

    # User inputs for architecture
    col1, col2 = st.columns(2)
    with col1:
        num_layers = st.slider("Number of Hidden Layers", 1, 5, 2)
        neurons_per_layer = st.slider("Neurons per Layer", 4, 128, 16, step=4)
    with col2:
        activation = st.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])
        learning_rate = st.slider("Learning Rate", 0.001, 0.1, 0.01, step=0.001)

    # Generate synthetic data
    X, y, _ = generate_dataset(100, true_function_sin, 0.2, (-1.5, 1.5), seed=42)

    # Build and train the neural network
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class SimpleNN(nn.Module):
        def __init__(self, input_dim, output_dim, num_layers, neurons_per_layer, activation):
            super(SimpleNN, self).__init__()
            layers = []
            layers.append(nn.Linear(input_dim, neurons_per_layer))
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
                if activation == "ReLU":
                    layers.append(nn.ReLU())
                elif activation == "Tanh":
                    layers.append(nn.Tanh())
                elif activation == "Sigmoid":
                    layers.append(nn.Sigmoid())
            layers.append(nn.Linear(neurons_per_layer, output_dim))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    # Initialize the model
    model = SimpleNN(1, 1, num_layers, neurons_per_layer, activation)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train the model
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    epochs = 500
    losses = []
    # for epoch in range(epochs):
        # st.write(f"Epoch {epoch + 1}/{epochs}")  # Accessing and displaying the epoch
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Plot training loss
    st.subheader("Training Loss")
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss Curve")
    st.pyplot(fig)

    # Visualize predictions
    st.subheader("Model Predictions")
    X_test = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_pred = model(X_test_tensor).detach().numpy()

    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Training Data", color="blue", alpha=0.6)
    ax.plot(X_test, y_pred, label="NN Predictions", color="red")
    ax.set_title("Neural Network Predictions")
    ax.legend()
    st.pyplot(fig)

def interactive_regularization_demo():
    st.header("📉 Interactive Regularization Demo")
    st.write("Explore how L1 (Lasso) and L2 (Ridge) regularization affect model performance and coefficients.")

    # User inputs
    regularization_type = st.selectbox("Regularization Type", ["L1 (Lasso)", "L2 (Ridge)"])
    alpha = st.slider("Regularization Strength (Alpha)", 0.01, 10.0, 1.0, step=0.01)

    # Generate synthetic data
    X, y, _ = generate_dataset(100, true_function_quadratic, 0.5, (-1.5, 1.5), seed=42)

    # Fit model
    from sklearn.linear_model import Lasso, Ridge

    if regularization_type == "L1 (Lasso)":
        model = Lasso(alpha=alpha)
    else:
        model = Ridge(alpha=alpha)

    model.fit(X, y)
    y_pred = model.predict(X)

    # Plot results
    st.subheader("Model Predictions")
    fig, ax = plt.subplots()
    ax.scatter(X, y, label="Training Data", color="blue", alpha=0.6)
    ax.plot(X, y_pred, label=f"{regularization_type} Predictions", color="red")
    ax.set_title(f"{regularization_type} Regularization")
    ax.legend()
    st.pyplot(fig)

    # Display coefficients
    st.subheader("Model Coefficients")
    st.write(f"Coefficients: {model.coef_}")

def custom_data_generation():
    st.header("✏️ Custom Data Generation")
    st.write("Draw your own data points and see how models fit them!")

    # User input for drawing data
    st.write("Add data points manually using the input fields below.")
    if "custom_data_points" not in st.session_state:
        st.session_state["custom_data_points"] = []

    # Input fields for adding points
    col1, col2 = st.columns(2)
    with col1:
        x_point = st.number_input("X-coordinate", value=0.0, step=0.1)
    with col2:
        y_point = st.number_input("Y-coordinate", value=0.0, step=0.1)

    if st.button("Add Point"):
        st.session_state["custom_data_points"].append((x_point, y_point))

    if st.button("Clear Data"):
        st.session_state["custom_data_points"] = []

    # Plot for drawing
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_title("Custom Data Points")
    if st.session_state["custom_data_points"]:
        ax.scatter(*zip(*st.session_state["custom_data_points"]), color="blue", label="Data Points")
    ax.legend()
    st.pyplot(fig)

    # Fit a model if data exists
    if len(st.session_state["custom_data_points"]) > 1:
        X = np.array([p[0] for p in st.session_state["custom_data_points"]]).reshape(-1, 1)
        y = np.array([p[1] for p in st.session_state["custom_data_points"]])

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        # Plot predictions
        st.subheader("Model Fit")
        fig, ax = plt.subplots()
        ax.scatter(X, y, label="Data Points", color="blue")
        ax.plot(X, y_pred, label="Linear Fit", color="red")
        ax.legend()
        st.pyplot(fig)

def noise_type_exploration():
    st.header("🔊 Noise Type Exploration")
    st.write("Explore how different types of noise affect model performance.")

    # User inputs
    noise_type = st.selectbox("Noise Type", ["Gaussian", "Uniform", "Outliers"])
    noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5, step=0.1)

    # Generate data with noise
    X = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)
    y_true = true_function_sin(X.flatten())

    if noise_type == "Gaussian":
        noise = np.random.normal(0, noise_level, size=y_true.shape)
    elif noise_type == "Uniform":
        noise = np.random.uniform(-noise_level, noise_level, size=y_true.shape)
    else:  # Outliers
        noise = np.random.normal(0, noise_level, size=y_true.shape)
        noise[np.random.choice(len(noise), size=len(noise) // 5, replace=False)] *= 5

    y_noisy = y_true + noise

    # Plot data
    st.subheader("Noisy Data")
    fig, ax = plt.subplots()
    ax.plot(X, y_true, label="True Function", color="green")
    ax.scatter(X, y_noisy, label="Noisy Data", color="blue", alpha=0.6)
    ax.legend()
    st.pyplot(fig)

def cross_validation_visualization():
    st.header("🔄 Cross-Validation Visualization")
    st.write("See how cross-validation splits data and affects model performance.")

    # User inputs
    k_folds = st.slider("Number of Folds", 2, 10, 5)

    # Generate data
    X, y, _ = generate_dataset(100, true_function_sin, 0.2, (-1.5, 1.5), seed=42)

    # Perform cross-validation
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fig, ax = plt.subplots()

    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        ax.scatter(X[train_idx], y[train_idx], label=f"Fold {i+1} Train", alpha=0.6)
        ax.scatter(X[test_idx], y[test_idx], label=f"Fold {i+1} Test", alpha=0.6)

    ax.set_title("Cross-Validation Splits")
    ax.legend()
    st.pyplot(fig)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Neural Network Playground", 
    "Regularization Demo", 
    "Custom Data Generation", 
    "Noise Exploration", 
    "Cross-Validation"
])

with tab1:
    neural_network_playground()

with tab2:
    interactive_regularization_demo()

with tab3:
    custom_data_generation()

with tab4:
    noise_type_exploration()
    X_noisy = np.linspace(-1.5, 1.5, 100).reshape(-1, 1)
    y_true_noisy = true_function_quadratic(X_noisy.flatten())
    noise_type = st.selectbox("Noise Type", ["Gaussian", "Uniform", "Outliers"], key="noise_type_selectbox")
    noise_level = st.slider("Noise Level", 0.1, 2.0, 0.5, step=0.1, key="noise_level_slider")

    if noise_type == "Gaussian":
        noise = np.random.normal(0, noise_level, size=y_true_noisy.shape)
    elif noise_type == "Uniform":
        noise = np.random.uniform(-noise_level, noise_level, size=y_true_noisy.shape)
    else:  # Outliers
        noise = np.random.normal(0, noise_level, size=y_true_noisy.shape)
        noise[np.random.choice(len(noise), size=len(noise) // 5, replace=False)] *= 5

    y_noisy = y_true_noisy + noise
    model = LinearRegression()
    model.fit(X_noisy, y_noisy)
    y_pred_noisy = model.predict(X_noisy)

    # Display model details
    st.subheader("Model Details")
    st.write("Coefficients:", model.coef_[0])
    st.write("Intercept:", model.intercept_)

    # Plot predictions
    st.subheader("Model Fit on Noisy Data")
    fig, ax = plt.subplots()
    ax.plot(X_noisy, y_true_noisy, label="True Function", color="green")
    ax.scatter(X_noisy, y_noisy, label="Noisy Data", color="blue", alpha=0.6)
    ax.plot(X_noisy, y_pred_noisy, label="Linear Fit", color="red")
    ax.legend()
    st.pyplot(fig)

with tab5:
    cross_validation_visualization()

# Create focused tabbed interface for learning resources
tab1, tab2, tab3 = st.tabs([
    "📊 Core Theory", 
    "🔬 Advanced Topics", 
    "📚 Research Resources"
])

with tab1:
    st.header("📊 Bias-Variance Theory & Fundamentals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("The Bias-Variance Decomposition")
        st.markdown("""
        **Mathematical Foundation:**
        ```
        Expected Test Error = Bias² + Variance + Irreducible Error
        ```
        
        **Key Components:**
        - **Bias²**: Systematic error from model simplification
        - **Variance**: Error from sensitivity to training data
        - **Irreducible Error**: Inherent noise (σ²)
        
        **Interpretation:** Optimal models balance bias and variance to minimize total error, not individual components.
        """)
        
        st.subheader("VC Dimension & Statistical Learning")
        st.markdown("""
        **Definition:** Maximum number of points that can be shattered by the hypothesis class.
        
        **Examples:**
        - Linear classifier in 2D: VC = 3
        - Polynomial degree d: VC ≈ d + 1
        
        **Generalization Bound:**
        ```
        Generalization Error ≤ Training Error + 
        √((VC_dim * log(n) + log(1/δ)) / n)
        ```
        
        **Implication:** Higher VC dimension requires more training data for reliable generalization.
        """)
    
    with col2:
        st.subheader("Model Complexity Spectrum")
        st.markdown("""
        **Complexity Progression:**
        1. **Linear Models** → High bias, low variance
        2. **Polynomial Models** → Flexible, increasing variance
        3. **Tree Methods** → Non-linear capability, high variance risk
        4. **Ensemble Methods** → Variance reduction through aggregation
        5. **Neural Networks** → Highly flexible, require regularization
        
        **Complexity Control:**
        - Regularization (L1/L2)
        - Early stopping
        - Pruning & dropout
        - Cross-validation
        """)
        
        st.subheader("Learning Curves & Data Efficiency")
        st.markdown("""
        **Key Patterns:**
        - **Small Data**: Variance dominates error
        - **Medium Data**: Bias-variance tradeoff emerges
        - **Large Data**: Bias becomes limiting factor
        
        **Convergence Properties:**
        - Parametric models: Error decreases as O(1/√n)
        - Non-parametric: Depends on intrinsic dimension
        
        **Data Strategy:** More data reduces variance but not bias; feature engineering addresses bias.
        """)

with tab2:
    st.header("🔬 Advanced Topics in Statistical Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Modern Perspectives")
        st.markdown("""
        **Double Descent Phenomenon:**
        - Classical view challenged by overparameterization
        - Test error shows secondary decrease beyond interpolation threshold
        
        **Neural Network Insights:**
        - Implicit regularization during optimization
        - Lottery ticket hypothesis
        - Neural tangent kernel theory
        
        **Ensembles & Variance Reduction:**
        ```
        Variance_ensemble = ρσ² + (1-ρ)σ²/M
        ```
        where ρ is correlation between learners, M is ensemble size.
        """)
        
        st.subheader("Regularization Theory")
        st.markdown("""
        **Forms & Effects:**
        
        **L2 (Ridge):**
        ```
        L(θ) = MSE(θ) + λ||θ||₂²
        ```
        - Shrinks weights proportionally
        - Reduces variance, may increase bias
        
        **L1 (Lasso):**
        ```
        L(θ) = MSE(θ) + λ||θ||₁
        ```
        - Induces sparsity
        - Performs feature selection
        
        **Modern Techniques:**
        - Dropout as ensemble method
        - Batch normalization effects
        - Early stopping as implicit regularization
        """)
    
    with col2:
        st.subheader("Concentration Inequalities")
        st.markdown("""
        **Key Results:**
        
        **Hoeffding's Inequality:**
        ```
        P(|X̄ - E[X]| ≥ t) ≤ 2exp(-2nt²/(b-a)²)
        ```
        
        **Rademacher Complexity:**
        ```
        R̂ₙ(F) = E[sup_{f∈F} (1/n)Σᵢσᵢf(xᵢ)]
        ```
        
        **Generalization Bound:**
        ```
        |R(f) - R̂(f)| ≤ 2R̂ₙ(F) + √(log(1/δ)/(2n))
        ```
        
        **Application:** Provides theoretical guarantees for generalization performance.
        """)
        
        st.subheader("Domain-Specific Applications")
        st.markdown("""
        **Computer Vision:**
        - Deep models necessary for image complexity
        - Data augmentation critical for variance reduction
        - Transfer learning reduces effective VC dimension
        
        **Natural Language Processing:**
        - Pretrained models reduce bias significantly
        - Fine-tuning balances transfer vs. overfitting
        - Attention mechanisms provide adaptive complexity
        
        **Time Series:**
        - Stationarity affects bias-variance tradeoff
        - Ensemble methods particularly effective
        - Online learning handles concept drift
        """)

with tab3:
    st.header("📚 Research Resources & Practical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Seminal Literature")
        st.markdown("""
        **Foundational Texts:**
        - Hastie et al. "Elements of Statistical Learning"
        - Bishop. "Pattern Recognition and Machine Learning"
        - Shalev-Shwartz & Ben-David. "Understanding Machine Learning"
        
        **Key Research Papers:**
        - Geman et al. (1992). "Neural Networks and the Bias/Variance Dilemma"
        - Belkin et al. (2019). "Reconciling Modern ML and the Bias-Variance Trade-off"
        - Zhang et al. (2017). "Understanding Deep Learning Requires Rethinking Generalization"
        """)
        
        st.subheader("Research Directions")
        st.markdown("""
        **Open Questions:**
        - Theoretical explanations for double descent
        - Generalization in overparameterized models
        - Information-theoretic bounds for neural networks
        - Sample efficiency in deep reinforcement learning
        
        **Emerging Areas:**
        - Self-supervised learning
        - Few-shot and meta-learning
        - Calibration and uncertainty quantification
        - Causal representation learning
        """)
    
    with col2:
        st.subheader("Experimental Design")
        st.markdown("""
        **Best Practices:**
        - Systematic parameter exploration
        - Multiple random seeds (50-100 minimum)
        - Confidence intervals for metrics
        - Baseline model comparison
        
        **Common Pitfalls:**
        - Data leakage in preprocessing
        - Insufficient cross-validation
        - Hyperparameter optimization bias
        - Overlooking statistical significance
        
        **Key Metrics:**
        - Training vs. validation curves
        - Bias-variance decomposition
        - Generalization gap analysis
        - Model capacity measures
        """)
        
        st.subheader("Implementation Tools")
        st.markdown("""
        **Core Libraries:**
        ```python
        # Essential packages
        scikit-learn    # Traditional ML algorithms
        pytorch        # Research-friendly DL framework
        tensorflow     # Production-ready platform
        
        # Specialized tools
        optuna         # Hyperparameter optimization
        ray[tune]      # Distributed experimentation
        captum         # Model interpretability
        mlflow         # Experiment tracking
        ```
        
        **Research Resources:**
        - Papers With Code (paperswithcode.com)
        - NeurIPS, ICML, ICLR proceedings
        - arXiv.org (cs.LG, stat.ML categories)
        - Distill.pub for visual explanations
        """)
