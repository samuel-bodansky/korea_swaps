curve_and_butterfly_strategies = {
    # --- Curve Strategies ---
    "2y10y Steepener": {'tenors': ['2y', '10y'], 'weights': [-1, 1]},
    "2y10y Flattener": {'tenors': ['2y', '10y'], 'weights': [1, -1]},
    "2y5y Steepener": {'tenors': ['2y', '5y'], 'weights': [-1, 1]},
    "2y5y Flattener": {'tenors': ['2y', '5y'], 'weights': [1, -1]},
    "1y5y Steepener": {'tenors': ['1y', '5y'], 'weights': [-1, 1]},
    "1y5y Flattener": {'tenors': ['1y', '5y'], 'weights': [1, -1]},
    "7y10y Steepener": {'tenors': ['7y', '10y'], 'weights': [-1, 1]},
    "7y10y Flattener": {'tenors': ['7y', '10y'], 'weights': [1, -1]},
    "7y15y Steepener": {'tenors': ['7y', '15y'], 'weights': [-1, 1]},
    "7y15y Flattener": {'tenors': ['7y', '15y'], 'weights': [1, -1]},
    "5y10y Steepener": {'tenors': ['5y', '10y'], 'weights': [-1, 1]},
    "5y10y Flattener": {'tenors': ['5y', '10y'], 'weights': [1, -1]},

    # --- Butterfly Strategies ---
    # Very Short End
    "3m6m1y Butterfly Long Belly": {'tenors': ['3m', '6m', '1y'], 'weights': [1, -2, 1]},
    "3m6m1y Butterfly Short Belly": {'tenors': ['3m', '6m', '1y'], 'weights': [-1, 2, -1]},
    "6m1y2y Butterfly Long Belly": {'tenors': ['6m', '1y', '2y'], 'weights': [1, -2, 1]},
    "6m1y2y Butterfly Short Belly": {'tenors': ['6m', '1y', '2y'], 'weights': [-1, 2, -1]},

    # Short-Mid Curve
    "1y2y5y Butterfly Long Belly": {'tenors': ['1y', '2y', '5y'], 'weights': [1, -2, 1]},
    "1y2y5y Butterfly Short Belly": {'tenors': ['1y', '2y', '5y'], 'weights': [-1, 2, -1]},
    "2y3y5y Butterfly Long Belly": {'tenors': ['2y', '3y', '5y'], 'weights': [1, -2, 1]},
    "2y3y5y Butterfly Short Belly": {'tenors': ['2y', '3y', '5y'], 'weights': [-1, 2, -1]},
    "3y4y6y Butterfly Long Belly": {'tenors': ['3y', '4y', '6y'], 'weights': [1, -2, 1]},
    "3y4y6y Butterfly Short Belly": {'tenors': ['3y', '4y', '6y'], 'weights': [-1, 2, -1]},
    "3y5y7y Butterfly Long Belly": {'tenors': ['3y', '5y', '7y'], 'weights': [1, -2, 1]},
    "3y5y7y Butterfly Short Belly": {'tenors': ['3y', '5y', '7y'], 'weights': [-1, 2, -1]},
    "4y5y8y Butterfly Long Belly": {'tenors': ['4y', '5y', '8y'], 'weights': [1, -2, 1]},
    "4y5y8y Butterfly Short Belly": {'tenors': ['4y', '5y', '8y'], 'weights': [-1, 2, -1]},

    # Mid-Long Curve
    "6y8y10y Butterfly Long Belly": {'tenors': ['6y', '8y', '10y'], 'weights': [1, -2, 1]},
    "6y8y10y Butterfly Short Belly": {'tenors': ['6y', '8y', '10y'], 'weights': [-1, 2, -1]},
    "7y10y20y Butterfly Long Belly": {'tenors': ['7y', '10y', '20y'], 'weights': [1, -2, 1]},
    "7y10y20y Butterfly Short Belly": {'tenors': ['7y', '10y', '20y'], 'weights': [-1, 2, -1]},
    "8y10y12y Butterfly Long Belly": {'tenors': ['8y', '10y', '12y'], 'weights': [1, -2, 1]},
    "8y10y12y Butterfly Short Belly": {'tenors': ['8y', '10y', '12y'], 'weights': [-1, 2, -1]},
    "10y12y15y Butterfly Long Belly": {'tenors': ['10y', '12y', '15y'], 'weights': [1, -2, 1]},
    "10y12y15y Butterfly Short Belly": {'tenors': ['10y', '12y', '15y'], 'weights': [-1, 2, -1]},
}