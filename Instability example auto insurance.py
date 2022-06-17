import pandas as pd
import graphviz
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz

dataset = 'second dataset'
data = pd.DataFrame()

match dataset:

    case 'first dataset':

        # first dataset
        data['nyears'] = [1, 2, 2, 4, 8]
        data['sports car'] = [True, False, True, False, False]
        data['risk'] = ['high risk', 'high risk', 'high risk', 'low risk', 'low risk']

    case 'second dataset':

        # second dataset
        data['nyears'] = [1, 4, 2, 4, 8]
        data['sports car'] = [True, True, True, False, False]
        data['risk'] = ['high risk', 'high risk', 'high risk', 'low risk', 'low risk']


# training
decision_tree = DecisionTreeClassifier(criterion='entropy')
decision_tree = decision_tree.fit(X=data[['nyears', 'sports car']], y=data['risk'])


# rendered tree (graphviz - pdf)
# dot_data = export_graphviz(
#     decision_tree=decision_tree, 
#     # out_file=f'rendered_trees/auto insurance - {dataset} - Copie',
#     out_file=None, 
#     feature_names=data[['nyears', 'sports car']].columns,
#     class_names=['high risk', 'low risk'],
#     filled=False,
#     rounded=True,
#     impurity=False,
#     proportion=False
#     )

# Draw graph
# graph = graphviz.Source(dot_data, format="pdf")
# graph.render(directory="rendered_trees", filename=f"auto insurance - {dataset}")

# graph = graphviz.Source.from_file('rendered_trees/auto insurance - first dataset - Copie')
# graph.render(directory="rendered_trees", filename=f"auto insurance - {dataset} - Copie")

graph = graphviz.Source.from_file('rendered_trees/auto insurance - second dataset dataset - Copie')
graph.render(directory="rendered_trees", filename=f"auto insurance - {dataset} - Copie")


# rendered tree (matplotlib - png)
# fig = plt.figure(figsize=(25,20))
# _ = plot_tree(
#     decision_tree=decision_tree,
#     feature_names=data[['nyears', 'sports car']].columns,
#     class_names=['high risk', 'low risk'],
#     filled=False
#     )

# fig.savefig(f"rendered_trees/auto insurance - {dataset}.png")





