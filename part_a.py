from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
import pandas as pd
import numpy as np
import pdb 
label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = pd.read_csv(file_name) 
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OrdinalEncoder()
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    #print(final_data)
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    #print(X.to_numpy())
    #print(y.to_numpy().flatten())
    return X.to_numpy(), y.to_numpy().flatten()

def entropy(y):
    #unique_y, counts_y = np.unique(y, return_counts= True)
    entropy = 0
    n = len(y)
    prob_y = np.sum(y) / n
    if prob_y > 0:
        entropy -= prob_y * np.log2(prob_y)
    if prob_y < 1:
        entropy -= (1-prob_y) * np.log2(1-prob_y )
    return entropy

def mi_score_cat(attribute, y):
    unique_attribute_values, counts_attribute = np.unique(attribute, return_counts=True)
    unique_y_values, counts_y = np.unique(y, return_counts=True)
    """if len(unique_attribute_values) == 1:
        return -np.inf"""
    mutual_info = 0.0
    mutual_info += entropy(y)
    """if len(counts_attribute) == 1:
        return mutual_info"""
    for value_attr in unique_attribute_values:
        p_attr = counts_attribute[unique_attribute_values == value_attr] / len(attribute)
        for value_y in unique_y_values:
            p_attr_and_y = np.sum(np.logical_and(attribute == value_attr, y == value_y)) / counts_attribute[unique_attribute_values == value_attr]
            #p_y = counts_y[unique_y_values == value_y] / len(y)
            if p_attr_and_y > 0:
                mutual_info += p_attr * p_attr_and_y * np.log2(p_attr_and_y)

    return mutual_info

def mi_score_cont(attribute, y):
    unique_attribute_values, counts_attribute = np.unique(attribute, return_counts=True)
    unique_y_values, counts_y = np.unique(y, return_counts=True)
    if len(unique_attribute_values) == 1:
        return -np.inf
    mutual_info = 0.0
    mutual_info += entropy(y)
    median = np.median(attribute)
    #for value_attr in unique_attribute_values:
    """if len(counts_attribute) == 1:
        return mutual_info"""
    p_attr = np.sum(counts_attribute[unique_attribute_values <= median]) / len(attribute)
    p_attr_num = p_attr * len(attribute)
    for value_y in unique_y_values:
        p_attr_and_y = np.sum(np.logical_and(attribute <= median, y == value_y)) / p_attr_num
        #p_y = counts_y[unique_y_values == value_y] / len(y)
        #if p_attr_and_y > 0:
        mutual_info += p_attr * p_attr_and_y * np.log2(p_attr_and_y + 1e-10)
    p_attr = np.sum(counts_attribute[unique_attribute_values > median]) / len(attribute)
    p_attr_num = p_attr * len(attribute)
    #print(p_attr)
    for value_y in unique_y_values:
        p_attr_and_y = np.sum(np.logical_and(attribute > median, y == value_y)) / p_attr_num
        #p_y = counts_y[unique_y_values == value_y] / len(y)
        #if p_attr_and_y > 0:
        mutual_info += p_attr * p_attr_and_y * np.log2(p_attr_and_y + 1e-10)    
    return mutual_info

    """    unique_x_j, counts_x_j = np.unique(data, return_counts= True)
    unique_y, counts_y = np.unique(data_y, return_counts= True)
    mi = 0
    n = len(data)
    for value_y in unique_y:
        prob_y = counts_y[unique_y == value_y] / n
        if prob_y > 0:
            mi -= prob_y * np.log2(prob_y)
    for x_j in unique_x_j:
        prob_x_j = counts_x_j[unique_x_j == x_j] / n
        for value_y in unique_y:
            prob_x_j_and_y = np.sum(np.logical_and(data == x_j, data_y == value_y)) / (prob_x_j * n)
            prob_y = counts_y[unique_y == value_y] / n
            if prob_x_j_and_y > 0:
                mi += prob_x_j*prob_y*prob_x_j_and_y * np.log2(prob_x_j_and_y)
    return mi"""

class DTNode:

    def __init__(self, depth, is_leaf = False, value = 0, column = None, split_value = None):

        #to split on column
        self.depth = depth
        #self.type = None
        #add children afterwards
        self.children = None
        self.split_value = split_value
        #if leaf then also need value
        self.is_leaf = is_leaf
        self.value = value
        self.parent = None
        if(not self.is_leaf):
            self.column = column


    def get_children(self, X, y, types):
        '''
        Args:
            X: A single example np array [num_features]
        Returns:
            child: A DTNode
        '''
        """if np.sum(y == 1) == 1 and np.sum(y == 0) == 1:
            self.is_leaf = True
            self.value = 1
            return [], None"""
        if self.is_leaf:
            return [], None
        num_col = X.shape[1]
        #print(num_col)
        n = X.shape[0]
        if n == 0:
            return [], None
        best_col = None
        max_mi = -1
        for j in range(num_col):
            if types[j] == "cat":
                # Categorical attribute
                mi = mi_score_cat(X[:,j], y)
            else:
                # for continuous attribute
                median = np.median(X[:, j])
                left_child = X[:, j] <= median
                #print(left_child)
                #print(j)
                #print(X[left_child][:,j])
                right_child = X[:, j] > median
                if median == np.min(X[:, j]) and median == np.max(X[:, j]):
                    continue
                else:
                    mi = mi_score_cont(X[:, j], y)
        
            if mi > max_mi:
                max_mi = mi
                best_col = j
        #self.value = np.bincount(y).argmax()
        if np.sum(y)/len(y) >= 0.5:
            self.value = 1
        else:
            self.value = 0
        # l1 is the list to store the values of j for which 
        l1 = []
        if best_col != None:
            self.column = best_col
            #print(best_col)
            l = []
            if types[best_col] == "cat":
                for value in np.unique(X[:, best_col]):
                    if np.sum(y[X[:, best_col] == value])/len(y[X[:, best_col] == value]) > 0.5:
                        self.value = 1
                    else:
                        self.value = 0
                    l.append(DTNode(self.depth + 1, False, self.value, None, value))
            else:
                median = np.median(X[:, best_col])
                #print(median)
                left_child = X[:, best_col] <= median
                right_child = X[:, best_col] > median
                #print(left_child)
                #print(right_child)
                if True in left_child:
                    if np.sum(y[X[:, best_col] <= median])/len(y[X[:, best_col] <= median]) >= 0.5:
                        self.value = 1
                    else:
                        self.value = 0
                    l.append(DTNode(self.depth + 1, False, self.value, None, median))
                if True in right_child:
                    #print(right_child)
                    #print(np.bincount(y[right_child]))
                    if np.sum(y[X[:, best_col] > median])/len(y[X[:, best_col] > median]) >= 0.5:
                        self.value = 1
                    else:
                        self.value = 0
                    l.append(DTNode(self.depth + 1, False, self.value, None, median))
        else:
            self.column = 1
            l = []
            self.is_leaf = True
        #print(X[:, best_col])
        #print(l)
        return l, best_col

class DTTree:
    def __init__(self):
        #Tree root should be DTNode
        self.root = DTNode(0)       

    def fit(self, X, y, types, max_depth = 10):
        '''
        Makes decision tree
        Args:
            X: numpy array of data [num_samples, num_features]
            y: numpy array of classes [num_samples, 1]
            types: list of [num_features] with types as: cat, cont
                eg: if num_features = 4, and last 2 features are continious then
                    types = ['cat','cat','cont','cont']
            max_depth: maximum depth of tree
        Returns:
            None
        '''
        self.root = self.DTRec(X, y, types, max_depth, 0, self.root)

    def DTRec(self, X, y, types, max_depth, depth, node):
        """if np.sum(y == 1) == 1 and np.sum(y == 0) == 1:
            if np.sum(y)/len(y) >= 0.5:
                node.value = 1
            else:
                node.value = 0
            node.children = []
            node.column = 1
            node.children.append(DTNode(depth + 1, True, 0, 1))
            node.children.append(DTNode(depth + 1, True, 1, 1))
            return node
        """
        if depth >= max_depth or len(np.unique(y)) == 1:
            if depth > 9:
                #pdb.set_trace()
                print(depth, y)
            if np.sum(y)/len(y) >= 0.5:
                node.value = 1
            else:
                node.value = 0
            #print(node.value)
            node.is_leaf = True
            #print(node.depth)
            #return node
        else:
            node.depth = depth
            node.children, node.column = node.get_children(X, y, types)
            #node.type = types[node.column]
            #print(node.value)
            if node.children != []:
                if types[node.column] == "cat":
                    for child in node.children:
                        child_X = X[X[:, node.column] == child.split_value]
                        child_y = y[X[:, node.column] == child.split_value]
                        child.parent = node
                        child = self.DTRec(child_X, child_y, types, max_depth, depth + 1, child)
                else:
                    child = node.children[0]
                    child_X = X[X[:, node.column] <= child.split_value]
                    child_y = y[X[:, node.column] <= child.split_value]
                    if len(child_X) > 0:
                        child.parent = node
                        child = self.DTRec(child_X, child_y, types, max_depth, depth + 1, child)
                    if len(node.children) > 1:
                        child = node.children[1]
                        child_X = X[X[:, node.column] > child.split_value]
                        child_y = y[X[:, node.column] > child.split_value]
                        if len(child_X) > 0:
                            child.parent = node
                            child = self.DTRec(child_X, child_y, types, max_depth, depth + 1, child)
        return node

    def predict(self, node, data):
        '''
        Predicted classes for X
        Args:
            X: numpy array of data [num_samples, num_features]
        Returns:
            y: [num_samples, 1] predicted classes
        '''
        if node.is_leaf:
            #print(node.depth)
            return node.value
        else: 
            if node.children != None:
                if types[node.column] == "cat":
                    for child in node.children:
                        if data[node.column] == child.split_value:
                            return self.predict(child, data)
                    return node.value
                else:
                    if data[node.column] <= node.children[0].split_value:
                        return self.predict(node.children[0], data)
                    else:
                        if len(node.children) > 1:
                            return self.predict(node.children[1], data)
            #print(node.depth)
            #else:
            #return node.value
                    
    def post_prune(self, node, X_val, y_val):
        # Calculate accuracy without pruning
        original_accuracy = self.calculate_accuracy(X_val, y_val)

        # Temporarily convert this node into a leaf
        is_leaf_temp = node.is_leaf
        node.is_leaf = True
        node.children = None

        # Calculate accuracy after pruning
        pruned_accuracy = self.calculate_accuracy(X_val, y_val)

        # Recursively prune child nodes
        if node.children != None:
            for child in node.children:
                self.post_prune(child, X_val, y_val)
        
        # If pruning improves accuracy, keep the node pruned
        if pruned_accuracy >= original_accuracy:
            node.is_leaf = True
            node.children = None
        else:
            # Restore the original state (undo pruning)
            node.is_leaf = is_leaf_temp


    def calculate_accuracy(self, X, y):
        prediction = 0
        n = len(X)
        for i in range(n):
            if self.predict(self.root, X[i]) == y[i]:
                prediction += 1
        return (prediction / n) * 100


if __name__ == '__main__':
    
    #change the path if you want
    X_train, y_train = get_np_array('train.csv')
    #X_train= X_train[:100]
    #y_train= y_train[:100]
    X_test, y_test = get_np_array("val.csv")
  
    types = ['cat','cat','cat',"cat","cat","cont","cat","cat","cat" ,"cont","cont" ,"cont"]
    max_depth = 21
    tree = DTTree()
    tree.fit(X_train,y_train,types, max_depth)
    #print(tree)
    #y_pred = tree.predict(X_test)
    #print(y_test)
    #print(y_pred)
    #print(y_train)
    n1 = len(X_train)
    #print(y_test.shape)
    #print(n1)
    accuracy = 0
    for j in range(n1):
        #print(y_test[j], tree.predict(tree.root, X_test[j]))
        if y_train[j] == tree.predict(tree.root, X_train[j]):
            accuracy += 1
        # else:
            #print(j, y_train[j], tree.predict(tree.root, X_train[j]))
    print(f"Acccuracy over train data with max_depth = {max_depth} is:", accuracy/n1 * 100)
    accuracy = 0
    n1 = len(X_test)
    for j in range(n1):
        #print(y_test[j], tree.predict(tree.root, X_test[j]))
        if y_test[j] == tree.predict(tree.root, X_test[j]):
            accuracy += 1
        # else:
            #print(j, y_test[j], tree.predict(tree.root, X_test[j]))
    print(f"Acccuracy over test data with max_depth = {max_depth} is:", accuracy/n1 * 100) 
    #print(X_train)

################# Part-(b) #####################

label_encoder = None

def one_hot(file_name):
    global label_encoder
    data = pd.read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = pd.DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = pd.concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()

#types = np.array(types)
#categorical_attr = types == "cat"
X_train_one_hot, y_train_one_hot = one_hot("train.csv")
# print(X_train.shape)
X_test_one_hot, y_test_one_hot = one_hot("test.csv")
while(len(types) != X_train_one_hot.shape[1]):
    types = ['cat'] + types
print(len(types))
max_depth = 55
tree = DTTree()
tree.fit(X_train_one_hot,y_train_one_hot,types, max_depth)
accuracy = 0
# n1 = len(X_test)
# for j in range(n1):
#     if y_test[j] == tree.predict(tree.root, X_test[j]):
#         accuracy += 1/n1
#     else:
#         print(y_test[j], tree.predict(tree.root, X_test[j]))
        
# print(f"Acccuracy over test data with max_depth = {max_depth} is:", accuracy * 100)
n1 = len(X_train_one_hot)
for j in range(n1):
    if y_train_one_hot[j] == tree.predict(tree.root, X_train_one_hot[j]):
        accuracy += 1
    # else:
        # print(y_test[j], tree.predict(tree.root, X_test[j]))
        
print(f"Acccuracy over train data with max_depth = {max_depth} is:", (accuracy/n1) * 100)

################# Part-(c) #######################

X_val, y_val = one_hot("val.csv")

tree.post_prune(tree.root, X_val, y_val)
n1 = len(X_test)
accuracy = 0
for j in range(n1):
    #print(y_train[j], tree.predict(tree.root, X_test[j]))
    if y_test[j] == tree.predict(tree.root, X_test[j]):
        accuracy += 1
print(f"Acccuracy over test data with max_depth = {max_depth} is:", (accuracy /n1) * 100)

################## Part-(d) #######################

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Lists to store results
max_depths = [15, 25, 35, 45]
train_accuracies_depth = []
test_accuracies_depth = []

ccp_alphas = [0.001, 0.01, 0.1, 0.2]
train_accuracies_alpha = []
test_accuracies_alpha = []

# Vary max depth
for depth in max_depths:
    DT_Tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=0)
    DT_Tree.fit(X_train, y_train)
    train_pred = DT_Tree.predict(X_train)
    #print(train_pred)
    #train_accuracy = accuracy_score(y_train, train_pred)
    n1 = len(y_train)
    train_accuracy = 0
    for j in range(n1):
        if y_train[j] == train_pred[j]:
            train_accuracy += 1/n1
    print(f"Accuracy over training data using sklearn library with depth = {depth} is", train_accuracy * 100)
    test_pred = DT_Tree.predict(X_test)
    #print(test_pred)
    #print(train_accuracy)
    #print(y_test.shape, test_pred.shape)
    n1 = len(y_test)
    test_accuracy = 0
    for j in range(n1):
        if y_test[j] == test_pred[j]:
            test_accuracy += 1/n1
    print(f"Accuracy over test data using sklearn library with depth = {depth} is", test_accuracy * 100)
    #test_accuracy = accuracy_score(y_test, test_pred)
    train_accuracies_depth.append(train_accuracy)
    test_accuracies_depth.append(test_accuracy)

# Vary ccp_alpha
for alpha in ccp_alphas:
    DT_Tree = DecisionTreeClassifier(criterion='entropy', random_state=0, ccp_alpha=alpha)
    DT_Tree.fit(X_train, y_train)
    train_pred = DT_Tree.predict(X_train)
    n1 = len(y_train)
    train_accuracy = 0
    for j in range(n1):
        if y_train[j] == train_pred[j]:
            train_accuracy += 1/n1
    print(f"Accuracy over training data using sklearn library with ccp_alpha = {alpha} is", train_accuracy * 100)
    test_pred = DT_Tree.predict(X_test)
    n1 = len(y_test)
    test_accuracy = 0
    for j in range(n1):
        if y_test[j] == test_pred[j]:
            test_accuracy += 1/n1
    print(f"Accuracy over test data using sklearn library with ccp_alpha = {alpha} is", test_accuracy * 100)
    train_accuracies_alpha.append(train_accuracy)
    test_accuracies_alpha.append(test_accuracy)

# Plot results
plt.figure(figsize=(12, 6))

# Plot results for varying max depth
plt.subplot(1, 2, 1)
plt.plot(max_depths, train_accuracies_depth, marker='o', label='Train Accuracy')
plt.plot(max_depths, test_accuracies_depth, marker='o', label='Test Accuracy')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Varying Max Depth')
plt.legend()

# Plot results for varying ccp_alpha
plt.subplot(1, 2, 2)
plt.plot(ccp_alphas, train_accuracies_alpha, marker='o', label='Train Accuracy')
plt.plot(ccp_alphas, test_accuracies_alpha, marker='o', label='Test Accuracy')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Varying ccp_alpha')
plt.legend()

plt.tight_layout()
plt.show()

###################### Part-(e) ########################

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the parameter grid for grid search
param_grid = {'n_estimators': [50, 150, 250, 350], 'max_features': [0.1, 0.3, 0.5, 0.7, 0.9], 'min_samples_split': [2, 4, 6, 8, 10]}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(oob_score=True, random_state=0)

# Perform a grid search
grid_search = GridSearchCV(rf_classifier, param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters from the grid search
best_params = grid_search.best_params_

# Train the Random Forest model with the best parameters
best_rf_classifier = RandomForestClassifier(oob_score=True, random_state=0, **best_params)
best_rf_classifier.fit(X_train, y_train)

# Report training accuracy
train_accuracy = best_rf_classifier.score(X_train, y_train)

# Report out-of-bag accuracy
oob_accuracy = best_rf_classifier.oob_score_

# Report test accuracy
test_pred = best_rf_classifier.predict(X_test)
test_accuracy = 0
n1 = len(y_test)
for j in range(n1):
    if y_test[j] == test_pred[j]:
        test_accuracy += 1/n1

# Report validation accuracy
val_pred = best_rf_classifier.predict(X_val)
val_accuracy = 0
n1 = len(y_val)
for j in range(n1):
    if y_val[j] == val_pred[j]:
        val_accuracy += 1/n1

print("Best Parameters:", best_params)
print("Training Accuracy:", train_accuracy * 100)
print("Out-of-Bag Accuracy:", oob_accuracy * 100)
print("Validation Accuracy:", val_accuracy * 100)
print("Test Accuracy:", test_accuracy * 100)
