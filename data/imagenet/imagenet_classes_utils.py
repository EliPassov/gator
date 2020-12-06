import os

import pandas as pd


class ImageNetDataHandler:
    def __init__(self, path):
        self.df = pd.read_csv(path, header=None, index_col=0)
        columns = ['other_id','name','details','num_subclasses','subclasses','hierarchy_inx','num_images']
        self.df.columns = columns
        self.df['subclasses'] = self.df['subclasses'].apply(lambda x: [] if (x == '[]' or 'double' in x)
                            else [int(x)] if ',' not in x else [int(a) for a in x[1:-1].split(',')])

    def get_all_leaf_subclass(self, class_code):
        subclasses = []

        if self.df.loc[class_code, 'num_subclasses'] == 0:
            subclasses = [class_code]
        else:
            for subclass in self.df.loc[class_code, 'subclasses']:
                subclasses.extend(self.get_all_leaf_subclass(subclass))

        return subclasses

    def get_class_name_or_names(self, class_code):
        if isinstance(class_code, int):
            class_code = [class_code]
        return self.df.ix[class_code]['name'].tolist()


if __name__=='__main__':
    path = '/home/eli/Eli/imagenet/meta_fixed_linebreaks.csv'
    handler = ImageNetDataHandler(path)
    sub_classes = handler.get_all_leaf_subclass(1815)
    names = handler.get_class_name_or_names(sub_classes)
    print(names)