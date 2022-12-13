from function import merge_group_avg
import pandas as pd

def test_merge_group_avg():
    
    # Create helper data
    helper1 = {'name': ['apple', 'banana', 'orange', 'pineapple'],
              'count': [5, 7, 3, 1],
               'colour': ['red', 'yellow', 'orange', 'yellow']}
    helper2 = {'name': ['apple', 'banana', 'orange', 'pineapple'],
               'ripeness': [5, 5, 3, 4]}
    
    helper_data1 = pd.DataFrame.from_dict(helper1)
    
    helper_data2 = pd.DataFrame.from_dict(helper2)
    
    test = merge_group_avg(helper1_data1, helper_data2, 'colour', 'ripeness')
    
    assert test.shape == (3, 2)
    assert test['ripeness'].sum() == 12.5
    assert test['ripeness'].mean() == 4.166666666666667
    assert list(test['colour']) == ['red', 'yellow', 'orange']