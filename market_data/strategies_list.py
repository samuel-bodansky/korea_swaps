curve_and_butterfly_strategies = {
    # --- Curve Strategies ---
    # Steepeners
    # "3m1y Steepener": {'tenors': ['3m', '1y'], 'weights': [-1, 1]},
    # "3m2y Steepener": {'tenors': ['3m', '2y'], 'weights': [-1, 1]},
    # "6m1y Steepener": {'tenors': ['6m', '1y'], 'weights': [-1, 1]},
    # "6m2y Steepener": {'tenors': ['6m', '2y'], 'weights': [-1, 1]},
    "1y2y Steepener": {'tenors': ['1y', '2y'], 'weights': [-1, 1]},
    "1y5y Steepener": {'tenors': ['1y', '5y'], 'weights': [-1, 1]},
    "1y10y Steepener": {'tenors': ['1y', '10y'], 'weights': [-1, 1]},
    "1y15y Steepener": {'tenors': ['1y', '15y'], 'weights': [-1, 1]},
    "2y5y Steepener": {'tenors': ['2y', '5y'], 'weights': [-1, 1]},
    "2y7y Steepener": {'tenors': ['2y', '7y'], 'weights': [-1, 1]},
    "2y10y Steepener": {'tenors': ['2y', '10y'], 'weights': [-1, 1]},
    "2y15y Steepener": {'tenors': ['2y', '15y'], 'weights': [-1, 1]},
    "2y20y Steepener": {'tenors': ['2y', '20y'], 'weights': [-1, 1]},
    "3y5y Steepener": {'tenors': ['3y', '5y'], 'weights': [-1, 1]},
    "3y10y Steepener": {'tenors': ['3y', '10y'], 'weights': [-1, 1]},
    "5y10y Steepener": {'tenors': ['5y', '10y'], 'weights': [-1, 1]},
    "5y15y Steepener": {'tenors': ['5y', '15y'], 'weights': [-1, 1]},
    "5y20y Steepener": {'tenors': ['5y', '20y'], 'weights': [-1, 1]},
    "7y10y Steepener": {'tenors': ['7y', '10y'], 'weights': [-1, 1]},
    "7y15y Steepener": {'tenors': ['7y', '15y'], 'weights': [-1, 1]},
    "7y20y Steepener": {'tenors': ['7y', '20y'], 'weights': [-1, 1]},
    "10y15y Steepener": {'tenors': ['10y', '15y'], 'weights': [-1, 1]},
    "10y20y Steepener": {'tenors': ['10y', '20y'], 'weights': [-1, 1]},
    "15y20y Steepener": {'tenors': ['15y', '20y'], 'weights': [-1, 1]},

    # Flatteners
    # "3m1y Flattener": {'tenors': ['3m', '1y'], 'weights': [1, -1]},
    # "3m2y Flattener": {'tenors': ['3m', '2y'], 'weights': [1, -1]},
    # "6m1y Flattener": {'tenors': ['6m', '1y'], 'weights': [1, -1]},
    # "6m2y Flattener": {'tenors': ['6m', '2y'], 'weights': [1, -1]},
    "1y2y Flattener": {'tenors': ['1y', '2y'], 'weights': [1, -1]},
    "1y5y Flattener": {'tenors': ['1y', '5y'], 'weights': [1, -1]},
    "1y10y Flattener": {'tenors': ['1y', '10y'], 'weights': [1, -1]},
    "1y15y Flattener": {'tenors': ['1y', '15y'], 'weights': [1, -1]},
    "2y5y Flattener": {'tenors': ['2y', '5y'], 'weights': [1, -1]},
    "2y7y Flattener": {'tenors': ['2y', '7y'], 'weights': [1, -1]},
    "2y10y Flattener": {'tenors': ['2y', '10y'], 'weights': [1, -1]},
    "2y15y Flattener": {'tenors': ['2y', '15y'], 'weights': [1, -1]},
    "2y20y Flattener": {'tenors': ['2y', '20y'], 'weights': [1, -1]},
    "3y5y Flattener": {'tenors': ['3y', '5y'], 'weights': [1, -1]},
    "3y10y Flattener": {'tenors': ['3y', '10y'], 'weights': [1, -1]},
    "5y10y Flattener": {'tenors': ['5y', '10y'], 'weights': [1, -1]},
    "5y15y Flattener": {'tenors': ['5y', '15y'], 'weights': [1, -1]},
    "5y20y Flattener": {'tenors': ['5y', '20y'], 'weights': [1, -1]},
    "7y10y Flattener": {'tenors': ['7y', '10y'], 'weights': [1, -1]},
    "7y15y Flattener": {'tenors': ['7y', '15y'], 'weights': [1, -1]},
    "7y20y Flattener": {'tenors': ['7y', '20y'], 'weights': [1, -1]},
    "10y15y Flattener": {'tenors': ['10y', '15y'], 'weights': [1, -1]},
    "10y20y Flattener": {'tenors': ['10y', '20y'], 'weights': [1, -1]},
    "15y20y Flattener": {'tenors': ['15y', '20y'], 'weights': [1, -1]},

    # --- Butterfly Strategies ---
    # Very Short End
    # "3m6m1y Butterfly Long Belly": {'tenors': ['3m', '6m', '1y'], 'weights': [1, -2, 1]},
    # "3m6m1y Butterfly Short Belly": {'tenors': ['3m', '6m', '1y'], 'weights': [-1, 2, -1]},
    # "6m1y2y Butterfly Long Belly": {'tenors': ['6m', '1y', '2y'], 'weights': [1, -2, 1]},
    # "6m1y2y Butterfly Short Belly": {'tenors': ['6m', '1y', '2y'], 'weights': [-1, 2, -1]},
    "1y2y3y Butterfly Long Belly": {'tenors': ['1y', '2y', '3y'], 'weights': [1, -2, 1]},
    "1y2y3y Butterfly Short Belly": {'tenors': ['1y', '2y', '3y'], 'weights': [-1, 2, -1]},

    # Short-Mid Curve
    "1y2y5y Butterfly Long Belly": {'tenors': ['1y', '2y', '5y'], 'weights': [1, -2, 1]},
    "1y2y5y Butterfly Short Belly": {'tenors': ['1y', '2y', '5y'], 'weights': [-1, 2, -1]},
    "2y3y5y Butterfly Long Belly": {'tenors': ['2y', '3y', '5y'], 'weights': [1, -2, 1]},
    "2y3y5y Butterfly Short Belly": {'tenors': ['2y', '3y', '5y'], 'weights': [-1, 2, -1]},
    "2y4y6y Butterfly Long Belly": {'tenors': ['2y', '4y', '6y'], 'weights': [1, -2, 1]},
    "2y4y6y Butterfly Short Belly": {'tenors': ['2y', '4y', '6y'], 'weights': [-1, 2, -1]},
    "3y5y7y Butterfly Long Belly": {'tenors': ['3y', '5y', '7y'], 'weights': [1, -2, 1]},
    "3y5y7y Butterfly Short Belly": {'tenors': ['3y', '5y', '7y'], 'weights': [-1, 2, -1]},
    "4y6y8y Butterfly Long Belly": {'tenors': ['4y', '6y', '8y'], 'weights': [1, -2, 1]},
    "4y6y8y Butterfly Short Belly": {'tenors': ['4y', '6y', '8y'], 'weights': [-1, 2, -1]},
    "5y7y10y Butterfly Long Belly": {'tenors': ['5y', '7y', '10y'], 'weights': [1, -2, 1]},
    "5y7y10y Butterfly Short Belly": {'tenors': ['5y', '7y', '10y'], 'weights': [-1, 2, -1]},

    # Mid-Long Curve
    "6y8y10y Butterfly Long Belly": {'tenors': ['6y', '8y', '10y'], 'weights': [1, -2, 1]},
    "6y8y10y Butterfly Short Belly": {'tenors': ['6y', '8y', '10y'], 'weights': [-1, 2, -1]},
    "7y10y12y Butterfly Long Belly": {'tenors': ['7y', '10y', '12y'], 'weights': [1, -2, 1]},
    "7y10y12y Butterfly Short Belly": {'tenors': ['7y', '10y', '12y'], 'weights': [-1, 2, -1]},
    "8y10y15y Butterfly Long Belly": {'tenors': ['8y', '10y', '15y'], 'weights': [1, -2, 1]},
    "8y10y15y Butterfly Short Belly": {'tenors': ['8y', '10y', '15y'], 'weights': [-1, 2, -1]},
    "10y12y15y Butterfly Long Belly": {'tenors': ['10y', '12y', '15y'], 'weights': [1, -2, 1]},
    "10y12y15y Butterfly Short Belly": {'tenors': ['10y', '12y', '15y'], 'weights': [-1, 2, -1]},
    "10y15y20y Butterfly Long Belly": {'tenors': ['10y', '15y', '20y'], 'weights': [1, -2, 1]},
    "10y15y20y Butterfly Short Belly": {'tenors': ['10y', '15y', '20y'], 'weights': [-1, 2, -1]},
    "12y15y20y Butterfly Long Belly": {'tenors': ['12y', '15y', '20y'], 'weights': [1, -2, 1]},
    "12y15y20y Butterfly Short Belly": {'tenors': ['12y', '15y', '20y'], 'weights': [-1, 2, -1]},
}