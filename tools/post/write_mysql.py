import pymysql
import tqdm

class MySQLHelper(object):
    def __init__(
        self, 
        host='openwriteposhq.mysql.rds.aliyuncs.com', 
        port=3350, 
        user='ruewrite', 
        password='write1107!@#$', 
        database='balance', 
        table_name='rx_bigmodel_pic'
    ):
        self.mydb = pymysql.connect(
            host=host, 
            port=port, 
            user=user, 
            password=password, 
            db=database, 
            charset='utf8'
        )
        self.mycursor = self.mydb.cursor()
        self.write_sql = f"insert into `{table_name}` (`sLabel`, `sImgURL`, `sModelVersion`, `sFeature`) VALUES (%s, %s, %s, %s)"
        self.del_sql = f"delete from {table_name}"
        self.read_sql = f"select * from {table_name}"

    def write_val2table(self, val):
        self.mycursor.execute(self.write_sql, val)
    
    def read_table(self):
        self.mycursor.execute(self.read_sql)
        return self.mycursor.fetchall()
    
    def del_table(self):
        self.mycursor.execute(self.del_sql)
        self.mydb.commit()

    def close_cursor(self):
        self.mydb.commit()
        self.mycursor.close()


def save_keeps2mysql(feats, labels, files, class_list):
    mysql = MySQLHelper()
    mysql.del_table()
    pbar = tqdm.tqdm(total=labels.shape[0])
    for label_index, filename, feat in zip(labels, files, feats):
        label = class_list[label_index]
        feat = ','.join(map(str, feat.tolist()))
        val = (label[1:], filename, '20231115', f'[{feat}]')
        mysql.write_val2table(val)
        pbar.update(1)
    del pbar
    mysql.close_cursor()


if __name__=="__main__":
    mysql = MySQLHelper()
    # val = ('1111', '/0/backflow/aa.jpg', '20231115', '0.11,0.22.033')  
    # mysql.write_val2table(val)
    data = mysql.read_table()
    # mysql.del_table()
    mysql.close_cursor()
    import pdb; pdb.set_trace()
    print(data)
