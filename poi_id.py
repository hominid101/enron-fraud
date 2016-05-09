#!/usr/bin/python
import sys
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

sys.path.append("../tools/")

def PlotFeatures(features, poi, pred, mark_poi=True, mark_pred=False, 
                 feature_names=['Feature 1', 'Feature 2']):

    ### plot each cluster with a different color--add more colors for
    ### drawing more than 4 clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(poi):
        if mark_pred:
            plt.scatter(features[ii][0], features[ii][1], 
                        color=colors[pred[ii]], alpha=0.5)
        elif not poi[ii]:
            plt.scatter(features[ii][0], features[ii][1], 
                        color='grey', alpha=0.5)
        if mark_poi and poi[ii]:
            plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
                
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.show()


def explore_enron_data(enron_data):
    print "================= Exploring Enron Data Set ===================="
    people_count = len(enron_data.keys())
    print "Enron data set includes", people_count, "people."
    #print "People names in Enron data set = ", enron_data.keys()
    
    feature_count = len(enron_data[enron_data.keys()[0]].keys())
    print feature_count, " features are collected on each person."
    #print "The features are: ", enron_data[enron_data.keys()[0]].keys()
    
    poi_count = len([person for person in enron_data if enron_data[person]["poi"]==True])
    print "Only ", poi_count, " people in the data set are marked as POIs"
    print "The POIs are: ", [person for person in enron_data if enron_data[person]["poi"]==True]
    
    print "Missing features are denoted as 'NaN'. Percentage of missing features for POIs and non-POIs..."
    
    for feature in enron_data[enron_data.keys()[0]].keys():
        nan_poi = len([person for person in enron_data if ((enron_data[person][feature] =='NaN') & (enron_data[person]["poi"]==True))])
        nan_pct_poi = 100.0 * nan_poi/poi_count
        nan_non_poi = len([person for person in enron_data if ((enron_data[person][feature] =='NaN') & (enron_data[person]["poi"]==False))])
        nan_pct_non_poi = 100.0 * nan_non_poi/(people_count - poi_count)
        pct_diff = nan_pct_poi - nan_pct_non_poi
        if nan_poi==0 or pct_diff > 30.0 or pct_diff < -30.0:
            print "    {0:.1f}% of POIs and {1:.1f}% of non-POIs have unknown {2}".format(nan_pct_poi, nan_pct_non_poi, feature)
    print "Total number of stocks belonging to James Prentice = ", enron_data["PRENTICE JAMES"]["total_stock_value"]
    print "Number of emails from Wesley Colwell to a POI = ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
    print "Total value of stock options exercised by Jeffrey Skilling = ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
    print "Total money taken home by Ken Lay = ", enron_data["LAY KENNETH L"]["total_payments"]
    print "Total money taken home by Jeffrey Skilling = ", enron_data["SKILLING JEFFREY K"]["total_payments"]
    print "Total money taken home by Andrew Fastow = ", enron_data["FASTOW ANDREW S"]["total_payments"]


def check_enron_outliers(data_dict):
    fname="enron_salary_outlier.png"
    features_list = ["poi", "salary", "exercised_stock_options"]
    #features_list = ["poi", "from_this_person_to_poi", "shared_receipt_with_poi"]

    data = featureFormat(data_dict, features_list)
    midx = data[:, 2].argmax()
    feature_2_max = max(data[:, 2])
    print "idx of max ", features_list[2], " = ", midx
    print "max " , features_list[2], " = ", feature_2_max, ", ", data[:, 2][midx]

    plt.subplot(1,2,1)
    colors=map(lambda x: 'red' if x else 'grey', data[:, 0])
    plt.scatter(data[:, 1], data[:, 2], s=40+data[:,0], c=colors, alpha=0.5, lw=0.)
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])

    # Now remove one outlier
    data_dict.pop("TOTAL", 0)
    data = featureFormat(data_dict, features_list)
    plt.subplot(1,2,2)
    colors=map(lambda x: 'red' if x else 'grey', data[:, 0])
    plt.scatter(data[:, 1], data[:, 2], s=40+data[:,0], c=colors, alpha=0.5, lw=0.)
    #plt.ticklabel_format(axis([-0.2e7, 1.2e7, -0.5, 4.0])
    plt.ticklabel_format(useOffset=True)
    plt.xlabel(features_list[1])
    plt.ylabel(features_list[2])

    plt.title("{0} vs {1} Plots before and after Outlier Removal.".format(features_list[1], features_list[2]), x=-0.1, y=1.05)
    plt.show()
    #plt.savefig(fname)


# add a new feature called with_poi
def add_features(data_dict):
    my_dataset = {}
    for key in data_dict:
        features_dict = data_dict[key].copy()
        with_poi = features_dict["from_poi_to_this_person"]
        if features_dict["from_this_person_to_poi"] != 'NaN':
            if with_poi == 'NaN':
                with_poi == 0
            with_poi += features_dict["from_this_person_to_poi"]
        features_dict.update({'with_poi' : with_poi})
        my_dataset[key]=features_dict
    return my_dataset

def get_default_classifier(clf_name, classifiers):
    if clf_name == 'kNN (hand-tuned)':
        return KNeighborsClassifier(p=1, leaf_size=100)
    elif clf_name == 'Decision Tree (hand-tuned)':
        return DecisionTreeClassifier(min_samples_split=5, 
                                     min_samples_leaf=2)
    else:
        return classifiers[clf_name]

def validate_classifier(clf, dataset, features_list, folds = 1000, scale=False,
                        print_metrics=False):
    RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"
    METRICS_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    if scale:
        # Perform feature scaling - doesn't really help with KNN
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    precision = 0.0
    recall = 0.0
    f1 = 0.0
    f2 = 0.0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append(features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append(features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            else:
                true_positives += 1
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        if print_metrics:
            print clf
            print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)

        accuracy = 1.0*(true_positives + true_negatives)/total_predictions

        if true_positives > 0:
            total_identifications = true_positives + false_positives
            precision = 1.0*true_positives/total_identifications
            total_labels = true_positives+false_negatives
            recall = 1.0*true_positives/total_labels
            f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
            f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        if print_metrics:
            print METRICS_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
            print ""
    except:
        print "Got a divide by zero when trying out:", clf
    return accuracy, precision, recall, f1, f2

def print_feature_importances(clf_name, all_feature_names, all_features, 
                              all_labels, classifiers):
    #from sklearn.tree import ExtraTreeClassifier
    #clf = ExtraTreeClassifier()
    clf = get_default_classifier(clf_name, classifiers)
    clf.fit(all_features, all_labels)
    feature_importances = clf.feature_importances_
    sorted_importances = sorted(zip(all_feature_names, feature_importances),key=lambda x: x[1],reverse=True)
    print '''  <table class="answer">
    <tr>
      <th>Feature</th>
      <th>Importance</th>
    </tr>
    '''
    for tt in sorted_importances:
        print "    <tr>"
        print('      <td> {0} </td>'.format(tt[0]))
        print('      <td> {:0.3f} </td>'.format(tt[1]))
        print("    </tr>")
    print("  </table>")

def select_features(dataset, classifiers, clf_names):
    features_list = ['poi']
    all_feature_names = [key for key in dataset[dataset.keys()[0]].keys() 
                         if ((key != 'poi') and (key != 'email_address'))]
    print("All {0} Features: {1}".format(len(all_feature_names), 
                                         all_feature_names))
    features_list.extend(all_feature_names)
    data = featureFormat(dataset, features_list, sort_keys = True)
    all_labels, all_features = targetFeatureSplit(data)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    all_features_scaled = scaler.fit_transform(all_features)

    # First calculate the importance of each feature using a DT classifier
    print_feature_importances('Decision Tree', all_feature_names, 
                              all_features, all_labels, classifiers)

    feature_selection_history = dict()
    for clf_name in clf_names:
        clf = get_default_classifier(clf_name, classifiers)
        scale = True if clf_name in {'kNN', 'SVM', 'kNN (hand-tuned)'} else False
        performance = []
        best_precision = 0
        best_perf = 0.
        fcount = 3
        for fcount in range(2, 16):
            if fcount is 15:
                # Manual feature selection
                feature_names = ['salary', 'bonus', 'exercised_stock_options', 
                                 'with_poi']
            else:
                # K-best feature selection
                selector = SelectKBest(f_classif, k=fcount)
                if (scale==True):
                    selector.fit(all_features_scaled, all_labels)
                else:
                    selector.fit(all_features, all_labels)

                feature_names = [all_feature_names[i] for i in 
                                 selector.get_support(indices=True)]
            features_list = ['poi']
            features_list.extend(feature_names)
            accuracy, precision, recall, f1, f2 = validate_classifier(
                clf, dataset, features_list, scale=scale)
            performance.append([fcount, precision,recall,f1,f2,feature_names])
            if(best_perf <= f1):
                best_fcount = fcount
                best_perf = f1
        feature_selection_history[clf_name]=[performance, best_fcount]
    return feature_selection_history

def write_feature_selection_history(history):
    print '''  <table class="answer">
    <tr>
      <th rowspan="3">Selector</th>
      <th colspan="6">Classifier</th>
    </tr>
    <tr>
      <th colspan="3">KNN</th>
      <th colspan="3">Decision Tree</th>
    </tr>
    <tr>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
    '''
    for kk in range(14):
        print "    <tr>"
        if kk==13:
            print('      <td> Manual </td>')
        else:
            print('      <td> {:d} Best </td>'.format(kk+2))
        for clf_name in history.keys():
            for ii in range(1,4,1):
                perf = history[clf_name][0][kk][ii]
                if kk+2 == history[clf_name][1]:
                    print('      <td bgcolor="#00CC66">{:0.3f}</td>'.format(perf))
                else:
                    print('      <td>{:0.3f}</td>'.format(perf))
        print("    </tr>")
    print("  </table>")

    for clf_name in history.keys():
        best_kk = history[clf_name][1] - 2
        print("Selected {0} best features for {1} are {2}".format(best_kk+2, clf_name, history[clf_name][0][best_kk][5]))
        print("3 best features {0}".format(3, history[clf_name][0][1][5]))


def get_labels_n_features(clf_name, feature_selection_history):
    default_feature = 'Decision Tree (hand-tuned)'
    if clf_name == 'Decision Tree (hand-tuned)':
        feature_names = ['salary', 'bonus', 'exercised_stock_options', 
                         'with_poi']
    else:
        selection_data = feature_selection_history.get(
            clf_name, feature_selection_history[default_feature])
        best_kk = selection_data[1] - 2
        feature_names = selection_data[0][best_kk][5]
    features_list = ['poi']
    features_list.extend(feature_names)
    return features_list

def try_classifiers(feature_selection_history, dataset, classifiers):
    try_history = dict()
    for clf_name in classifiers:
        clf = get_default_classifier(clf_name, classifiers)
        features_list = get_labels_n_features(clf_name, 
                                               feature_selection_history)
        scale = True if clf_name in {'kNN', 'SVM', 'kNN (hand-tuned)'} else False
        accuracy, precision, recall, f1, f2 = validate_classifier(
            clf, dataset, features_list, scale=scale, print_metrics=True)
        try_history[clf_name]=[accuracy, precision,recall,f1,f2]
    return try_history

def write_classifier_try_history(history):
    print '''  <table class="answer">
    <tr>
      <th>Classifier</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>F2-score</th>
    </tr>
    '''
    for clf_name in history:
        print "    <tr>"
        print('      <td> {0} </td>'.format(clf_name))
        for ii in range(5):
            perf = history[clf_name][ii]
            if clf_name in {'kNN'}:
                print('      <td bgcolor="#00CC66">{:0.3f}</td>'.format(perf))
            else:
                print('      <td>{:0.3f}</td>'.format(perf))
        print("    </tr>")
    print("  </table>")


def tune_classifier(clf_name, clf, dataset, features_list, scores, folds = 1000):
    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    scale = True if clf_name in {'kNN', 'SVM', 'kNN (hand-tuned)'} else False
    if scale:
        # Perform feature scaling 
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        features = scaler.fit_transform(features)

    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    if clf_name == 'kNN':
        parameter_grid = [{'p': [1, 2, 3], 
                           'n_neighbors': [1, 5, 7, 10, 15],
                           'leaf_size': [30, 50, 70, 100]}]
    elif clf_name == 'Decision Tree':
        parameter_grid = [{'min_samples_split': [2, 3, 4, 5], 
                           'min_samples_leaf':[2, 3, 4, 5], 
                           'splitter': ['random', 'best']}]
    best_params={}
    for score in scores:
        grid_clf = GridSearchCV(clf, parameter_grid, cv=cv, 
                                scoring="{0}_weighted".format(score))
        grid_clf.fit(features, labels)
        best_params = grid_clf.best_params_
        #print("Grid scores:")
        #for params, mean_score, scores in grid_clf.grid_scores_:
        #    print("{:0.3f} {:+0.03f} for {!r}".format(mean_score, scores.std() * 2, params))
    print("Classifier {0} has tuned parameters {1}".format(clf_name, best_params))
    return best_params

def tune_classifiers(clf_names, feature_selection_history, my_dataset, 
                     classifiers):
    tuning_history=dict()
    best_clf_name=None
    best_perf=0.
    for clf_name in clf_names:
        features_list = get_labels_n_features(clf_name, 
                                               feature_selection_history)
        clf = get_default_classifier(clf_name, classifiers)
        if ((clf_name != 'kNN (hand-tuned)') and 
            (clf_name != 'Decision Tree (hand-tuned)')):
            params = tune_classifier(clf_name, clf, my_dataset, 
                                     features_list, scores=['f1'])
            clf.set_params(**params)
        scale = True if clf_name in {'kNN', 'SVM', 'kNN (hand-tuned)'} else False
        accuracy, precision, recall, f1, f2 = validate_classifier(
            clf, my_dataset, features_list, scale=scale)
        tuning_history[clf_name]=[accuracy, precision, recall, f1, f2, params]
        if(best_perf <= f1):
            best_clf_name=clf_name
            best_perf=f1
            #best_clf = clf
            #best_features_list = features_list
            #print("Best classifier {0} has accuracy {1}, precision {2}, recall {3}, F1 {4}".format(clf_name, accuracy, precision, recall, f1))
            #best_perf = f1
    return best_clf_name, tuning_history

def write_classifier_tuning_history(best_clf, history):
    print '''  <table class="answer">
    <tr>
      <th>Tuned Classifier</th>
      <th>Accuracy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>F2-score</th>
      <th>Tuned params</th>
    </tr>
    '''
    for clf_name in history:
        print "    <tr>"
        print('      <td> {0} </td>'.format(clf_name))
        for ii in range(5):
            perf = history[clf_name][ii]
            if clf_name == best_clf:
                print('      <td bgcolor="#00CC66">{:0.3f}</td>'.format(perf))
            else:
                print('      <td>{:0.3f}</td>'.format(perf))

        # Print parameters
        params = history[clf_name][5]
        if clf_name == best_clf:
            print('      <td bgcolor="#00CC66">{0}</td>'.format(params))
        else:
            print('      <td>{0}</td>'.format(params))
        print("    </tr>")
    print("  </table>")

def get_tuned_classifier(clf_name, classifiers, tuning_history):
    clf = get_default_classifier(clf_name, classifiers)
    if ((clf_name != 'kNN (hand-tuned)') and 
        (clf_name != 'Decision Tree (hand-tuned)')):
        params = tune_classifier(clf_name, clf, my_dataset, 
                                 features_list, scores=['f1'])
        clf.set_params(**params)
    return clf

##=========================================================================
##=========================================================================
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.grid_search import GridSearchCV

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

# Explore the data and print interesting information
print "Explore data"
explore_enron_data(data_dict)

### Task 2: Remove outliers
print "Task 1: Check and remove outliers"
check_enron_outliers(data_dict)
data_dict.pop("TOTAL", 0)

classifiers = {
    "kNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier()}

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
print "Task 3: Create new features and select the best set of features"
my_dataset = add_features(data_dict)

# Extract features and labels from dataset for local testing
clf_names=['kNN', 'Decision Tree (hand-tuned)']
feature_selection_history = select_features(my_dataset, 
                                            classifiers,
                                            clf_names=clf_names)
write_feature_selection_history(feature_selection_history)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
print "Task 4: Try a variety of classifiers and select two."
try_classifier_task = True
if try_classifier_task:
    classifier_try_history = try_classifiers(feature_selection_history, 
                                             my_dataset, classifiers)
    write_classifier_try_history(classifier_try_history)


### Task 5: Tune your classifier to achieve better than .3
### precision and recall using our testing script.  Because of the
### small size of the dataset, the script uses stratified shuffle
### split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print "Task 5: Tune the two best classifiers from previous step and select best."
best_perf=0.
clf_names = ['kNN', 'kNN (hand-tuned)', 'Decision Tree', 'Decision Tree (hand-tuned)']
clf_name, tuning_history = tune_classifiers(clf_names, feature_selection_history,
                                            my_dataset, classifiers)
write_classifier_tuning_history(clf_name, tuning_history)


# Submit the selected features and the tuned classifier with the best performance
features_list = get_labels_n_features(clf_name, feature_selection_history)
clf = get_tuned_classifier(clf_name, classifiers, tuning_history)
scale = True if clf_name in {'kNN', 'SVM', 'kNN (hand-tuned)'} else False
validate_classifier(clf, my_dataset, features_list, scale=scale, 
                    print_metrics=True)
### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.
dump_classifier_and_data(clf, my_dataset, features_list)
