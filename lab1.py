import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Solution:
    def __init__(self) -> None:
        # TODO:
        # Load data from data/chipotle.tsv file using Pandas library and
        # assign the dataset to the 'chipo' variable.
        file = 'data/chipotle.tsv'
        chipo = pd.read_csv(file, sep='\t')

        self.chipo = chipo

    def top_x(self, count) -> None:
        # TODO
        # Top x number of entries from the dataset and display as markdown format.
        topx = self.chipo.head(count)
        print(topx.to_markdown())

    def count(self) -> int:
        # TODO
        no_of_obser = self.chipo.shape[0]
        # The number of observations/entries in the dataset.
        return no_of_obser

    def info(self) -> None:
        # TODO
        return self.chipo.info

    def num_column(self) -> int:
        # TODO return the number of columns in the dataset
        return self.chipo.shape[1]

    def print_columns(self) -> None:
        return self.chipo.columns
        # TODO Print the name of all the columns.

    def most_ordered_item(self):
        # TODO
        df2 = pd.DataFrame()

        temp = self.chipo.groupby(['item_name']).sum()
        most_order_item_df = temp.sort_values('quantity', ascending=False)
        dumy_list = []
        df2 = most_order_item_df.head(1)

        dumy_list.append(most_order_item_df.head(1).index.values[0])
        dumy_list.append(most_order_item_df.head(1)["order_id"].values[0])
        dumy_list.append(most_order_item_df.head(1)["quantity"].values[0])

        return dumy_list[0], dumy_list[1], dumy_list[2]

    def total_item_orders(self) -> int:

        # TODO How many items were orderd in total?
        return self.chipo['quantity'].sum()

    def total_sales(self) -> float:

        # TODO
        print(type(self.chipo.item_price))
        self.chipo.info()
        if(self.chipo.dtypes['item_price'] != np.float64):
            self.chipo.item_price = self.chipo.item_price.apply(
                lambda x: float(x[1:]))                                 # lambda function(checks if the item_price is float, if not changes it to float)

        tot_sales = (self.chipo.quantity*self.chipo.item_price).sum()

        return tot_sales

    def num_orders(self) -> int:
        # TODO
        no_of_orders = self.chipo.order_id.max()
        # How many orders were made in the dataset?
        return no_of_orders

    def average_sales_amount_per_order(self) -> float:
        # TODO

        self.chipo["revenue"] = self.chipo["quantity"]*self.chipo["item_price"]
        ord_grp = order_grouped = self.chipo.groupby(by=['order_id']).sum()
        average_sales_amnt = order_grouped.mean()['revenue']
        return np.round(average_sales_amnt, 2)

    def num_different_items_sold(self) -> int:
        # TODO
        # How many different items are sold?
        return self.chipo.item_name.nunique()

    def plot_histogram_top_x_popular_items(self, x: int) -> None:
        from collections import Counter
        letter_counter = Counter(self.chipo.item_name)
        temp_list = list(letter_counter.items())
        temp_df = pd.DataFrame(temp_list, columns=[
            'itemnames', 'Number Of Orders'])
        temp_df.sort_values(by='Number Of Orders',
                            ascending=False, inplace=True)
        bar_df = temp_df.head(5)
        print(bar_df)
        items = bar_df["itemnames"]
        no_of_orders = bar_df["Number Of Orders"]
        plt.bar(items, no_of_orders, color='brown',
                width=0.4)

        plt.xlabel("ITEMS")
        plt.ylabel("NUMBER OF ORDERS")
        plt.title("Most popular items")
        plt.show(block=True)

    def scatter_plot_num_items_per_order_price(self) -> None:
        # TODO
        # list of prices by removing dollar sign
        price_list = self.chipo.item_price[1:].tolist()
        # removing trailing spaces
        temp_price_list = []
        for i in price_list:
            i = str(i)
            j = i.replace(' ', '')
            temp_price_list.append(j)
        orders = self.chipo.groupby('order_id').sum()
        x = orders.item_price
        y = orders.quantity
        s = 50
        c = "blue"
        plt.xlabel("order price")
        plt.ylabel("Items ordered(quantity)")
        plt.title("No of items ordered per order price")
        plt.scatter(x, y, s, c, alpha=0.5)
        plt.show()


def test() -> None:
    solution = Solution()
    solution.top_x(10)
    count = solution.count()

    assert count == 4622
    solution.info()
    count = solution.num_column()
    assert count == 5
    item_name, order_id, quantity = solution.most_ordered_item()

    assert item_name == 'Chicken Bowl'
    assert order_id == 713926
    assert quantity == 761
    total = solution.total_item_orders()
    assert total == 4972

    assert 39237.02 == solution.total_sales()

    assert 1834 == solution.num_orders()

    assert 21.39 == solution.average_sales_amount_per_order()
    assert 50 == solution.num_different_items_sold()
    solution.plot_histogram_top_x_popular_items(5)

    solution.scatter_plot_num_items_per_order_price()


if __name__ == "__main__":
    # execute only if run as a script
    test()
