################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.07.27
# filenaem: sql_server.py
# function: toolkit for reading or writing data from mysql database
######################################################
import pymysql
import tqdm
import datetime
import numpy as np


class MySQLHelper(object):
    def __init__(
        self,
        host="openwriteposhq.mysql.rds.aliyuncs.com",
        port=3350,
        user="ruewrite",
        password="write1107!@#$",
        database="balance",
        table_name="rx_bigmodel_pic",
        custom_keys=None,
    ):
        self.mydb = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            db=database,
            charset="utf8",
        )
        self.mycursor = self.mydb.cursor()
        self.write_sql_command = (
            f"insert into `{table_name}` (`sLabel`, `sImgURL`, `sModelVersion`, `sFeature`) VALUES (%s, %s, %s, %s)"
        )
        self.del_sql_command = f"delete from {table_name}"
        # self.del_sql = f"truncate table {table_name} " # insufficient permissions
        self.read_table_command = f"select {custom_keys or '*'} from {table_name}"
        self.read_column_names_command = f"SELECT * FROM {table_name} LIMIT 0"
        self.get_names_command = "select label,sgoodsname from vgoods"

    def write_val2table(self, val):
        self.mycursor.execute(self.write_sql_command, val)

    def read_table(self):
        self.mycursor.execute(self.read_table_command)
        return self.mycursor.fetchall()

    def read_names(self):
        self.mycursor.execute(self.get_names_command)
        return self.mycursor.fetchall()
    
    def read_column_names(self):
        self.mycursor.execute(self.read_column_names_command)
        return [desc[0] for desc in self.mycursor.description]

    def del_table(self):
        self.mycursor.execute(self.del_sql_command)
        self.mydb.commit()

    def close_cursor(self):
        self.mydb.commit()
        self.mycursor.close()


def create_sql_server(brand_id, custom_keys=None):
    assert brand_id is not None, f"please set a value for brand_id!"
    private_hostmap= {479: 1, 1000: 4, 684: 0}
    host_base = private_hostmap.get(brand_id, brand_id//100)
    return MySQLHelper(
        host=f"balance-open{host_base:02d}.mysql.rds.aliyuncs.com",
        port=3350,
        user="rueread",
        password="read1107!@#$",
        database=f"balance{brand_id}",
        table_name="tgoodscollectpic",
        custom_keys=custom_keys,
    )


def create_dates(set_dates):
    if len(set_dates) != 2:
        return set_dates
    def convert_datetypes(str_date):
        year, month, day = [int(num) for num in str_date.split("-")]
        return datetime.date(year, month, day)
    
    start_date = convert_datetypes(set_dates[0])
    end_date = convert_datetypes(set_dates[1])
    date_range = range((end_date - start_date).days + 1)
    date_list = [(start_date + datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in date_range]
    return date_list


def read_sql_data(sql_server, set_date, set_cats):
    assert set_date or set_cats is not None, "please set a value for date!"
    read_res = sql_server.read_table()
    gts, urls, product_ids, dates, preds = zip(*(read_res))
    product_ids, gts, urls, dates, preds = \
        np.array(product_ids), np.array(gts), np.array(urls), np.array(dates), np.array(preds)
    if set_cats is not None:
        keeps = np.isin(gts, set_cats)
    elif set_date is not None:
        assert len(set_date) <= 2, "the set_date must be a list with length 2 or 1!"
        keeps = np.isin(dates, create_dates(set_date))
    else:
        raise ValueError("Invalid set_cats or set_date!")
    assert keeps.shape[0] >= 1, "the sql data number must be greater than 0!"
    return product_ids[keeps], gts[keeps], urls[keeps], dates[keeps], preds[keeps]


def save_keeps2mysql(feats, labels, files, update_times=0):
    mysql = MySQLHelper()
    mysql.del_table()
    stride = labels.shape[0] // update_times if update_times else 1
    pbar = tqdm.tqdm(total=labels.shape[0], miniters=stride, maxinterval=3600)
    # 超过最长时间后会重新设置最长打印时间, 故设置为1h=3600s
    for i, (label_index, filename, feat) in enumerate(zip(labels, files, feats)):
        label = str(label_index)
        feat = ",".join(map(str, feat.tolist()))
        val = (label, filename, "20231115", f"[{feat}]")
        mysql.write_val2table(val)
        pbar.update(1)
        if not (i + 1) % stride:
            mysql.mydb.commit()
    pbar.close()
    mysql.close_cursor()


if __name__ == "__main__":
    mysql = MySQLHelper()
    # val = ('1111', '/0/backflow/aa.jpg', '20231115', '0.11,0.22.033')
    # mysql.write_val2table(val)
    data = mysql.read_table()
    # mysql.del_table()
    mysql.close_cursor()
    print(data)
