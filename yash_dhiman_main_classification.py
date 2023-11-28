
from yash_dhiman_classification import classification


import warnings
warnings.filterwarnings("ignore")


clf=classification("D:/IISER-B/IISERB Courses/5th Sem/ECS-308_DSML/Project/Phase_2", clf_opt='rf',
                        no_of_selected_features=18)

clf.classification()

