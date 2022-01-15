from ray import tune


TR1_hs_wd = {"batch_size": 512,
             "learning_rate": 0.000001,
             "weight_decay" : tune.grid_search([0.1, 0.001, 0.0001]),
             "h1_size" : tune.grid_search([100, 200, 300, 400, 500])}

TR2_1_hs_wd = {"batch_size": 1024,
               "learning_rate": 0.000001,
               "weight_decay" : 0.1,
               "h1_size" : tune.grid_search([200, 300, 400, 500, 600])}


TR2_2_hs_wd = {"batch_size": 1024,
               "learning_rate": 0.000001,
               "weight_decay" : 0.001,
               "h1_size" : tune.grid_search([200, 300, 400, 500, 600])}


TR2_3_hs_wd = {"batch_size": 1024,
               "learning_rate": 0.000001,
               "weight_decay" : 0.0001,
               "h1_size" : tune.grid_search([200, 300, 400, 500, 600])}