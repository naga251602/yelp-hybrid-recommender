"""
Method Description:
Hybrid recommendation system combining item-based collaborative filtering (CF)
with XGBoost regression. Features selected via greedy forward selection from EDA
on user.json, business.json, checkin.json, tip.json: business avg stars (json),
user avg stars (json), business latitude/longitude, user review count, business
price range, tip count per business/user, user compliments sum, user useful votes,
business category count, business review count, checkin count.

>=0 and <1 : 102226                                                                                                                                                                                                        
>=1 and <2 : 32861                                                                                                                                                                                                         
>=2 and <3 : 6179                                                                                                                                                                                                          
>=3 and <4 : 776                                                                                                                                                                                                           
>=4        : 2  

RMSE: 0.977267
Execution Time: 299.9s
"""

import sys
import json
import math
import time
import csv
from pyspark import SparkContext, SparkConf
from xgboost import XGBRegressor
from numpy import array as np_arr


class Tool:
    N = 40

    @staticmethod
    def cleaner(line):
        try:
            return json.loads(line)
        except:
            return None

    @staticmethod
    def pearson_sim(b_1, b_2, b_u):
        l_1, l_2 = b_u.get(b_1, {}), b_u.get(b_2, {})
        i_u = set(l_1.keys()) & set(l_2.keys())
        if len(i_u) < 2: return 0.0
        a_1 = sum(l_1[i] for i in i_u) / len(i_u)
        a_2 = sum(l_2[j] for j in i_u) / len(i_u)
        res = sum((l_1[k] - a_1) * (l_2[k] - a_2) for k in i_u)
        f_1 = math.sqrt(sum((l_1[k] - a_1) ** 2 for k in i_u))
        f_2 = math.sqrt(sum((l_2[k] - a_2) ** 2 for k in i_u))
        return 0.0 if (f_1 == 0 or f_2 == 0) else res / (f_1 * f_2)

    @staticmethod
    def read_csv_getRDD(context, input_file):
        raw_data = context.textFile(input_file)
        row_headers = raw_data.first()
        return raw_data.filter(lambda d: d != row_headers).map(lambda line: line.split(','))

    @staticmethod
    def read_json_getRDD(context, input_file):
        raw_data = context.textFile(input_file)
        return raw_data.map(lambda g: Tool.cleaner(g)).filter(lambda k: k is not None)

    @staticmethod
    def write_output(write_data, file):
        with open(file, 'w') as writer:
            for data in write_data:
                writer.write(data + "\n")


class Solution:
    def start(self):
        t_start  = time.time()
        input_fp = sys.argv[1]
        test_fp  = sys.argv[2]
        out_path = sys.argv[3]

        context = SparkContext(conf=SparkConf().setAppName("competition").setMaster("local[*]"))
        context.setLogLevel("ERROR")

        my_tr_rdd = Tool.read_csv_getRDD(context, input_fp + "/yelp_train.csv")
        my_tt_rdd = Tool.read_csv_getRDD(context, test_fp)
        b_rdd     = Tool.read_json_getRDD(context, input_fp + '/business.json')
        u_rdd     = Tool.read_json_getRDD(context, input_fp + '/user.json')
        ch_rdd    = Tool.read_json_getRDD(context, input_fp + '/checkin.json')
        tp_rdd    = Tool.read_json_getRDD(context, input_fp + '/tip.json')

        trrdd = my_tr_rdd.map(lambda g: (g[0], g[1], float(g[2])))
        ttrdd = my_tt_rdd.map(lambda g: (g[0], g[1]))

        b_u = (
            trrdd
            .map(lambda g: (g[1], (g[0], g[2])))
            .groupByKey()
            .mapValues(lambda k: dict(k))
            .collectAsMap()
        )
        u_b = (
            trrdd
            .map(lambda g: (g[0], (g[1], g[2])))
            .groupByKey()
            .mapValues(lambda k: dict(k))
            .collectAsMap()
        )

        g_avg   = trrdd.map(lambda g: g[2]).mean()
        avg_b_r = {b: sum(g.values()) / len(g) for b, g in b_u.items()}
        avg_u_r = {u: sum(g.values()) / len(g) for u, g in u_b.items()}

        feat_u = (
            u_rdd
            .map(lambda k: (k['user_id'], {
                "u_json_avg":  k.get("average_stars", 0.0),
                "u_rv_c":      k.get("review_count", 0),
                "useful":      k.get("useful", 0),
                "compliments": sum([
                    k.get("compliment_hot", 0), k.get("compliment_more", 0),
                    k.get("compliment_cool", 0), k.get("compliment_funny", 0),
                    k.get("compliment_writer", 0), k.get("compliment_photos", 0)
                ]),
            }))
            .collectAsMap()
        )
        feat_b = (
            b_rdd
            .map(lambda k: (k["business_id"], {
                "b_json_avg": k.get("stars", 0.0),
                "b_lat":      k.get("latitude", 0.0) if k.get("latitude") is not None else 0.0,
                "b_long":     k.get("longitude", 0.0) if k.get("longitude") is not None else 0.0,
                "b_rv_c":     k.get("review_count", 0),
                "b_price":    int(k.get("attributes", {}).get("RestaurantsPriceRange2") or 0)
                              if k.get("attributes") else 0,
                "b_cat":      len(k.get("categories", "").split(",")) if k.get("categories") else 0,
            }))
            .collectAsMap()
        )
        feat_ch = (
            ch_rdd
            .map(lambda k: (k["business_id"], sum(k["time"].values()) if isinstance(k.get("time"), dict) else 0))
            .collectAsMap()
        )
        t_b = (
            tp_rdd
            .map(lambda k: (k['business_id'], 1))
            .reduceByKey(lambda p, q: p + q)
            .collectAsMap()
        )
        t_u = (
            tp_rdd
            .map(lambda k: (k['user_id'], 1))
            .reduceByKey(lambda p, q: p + q)
            .collectAsMap()
        )

        def pr_rat(ui, bi):
            if ui not in u_b and bi not in b_u: return g_avg
            if ui not in u_b: return avg_b_r.get(bi, g_avg)
            if bi not in b_u: return avg_u_r.get(ui, g_avg)

            fb   = (avg_u_r[ui] + avg_b_r[bi]) / 2.0
            vals = []
            for o_b, r in u_b[ui].items():
                if o_b == bi: continue
                val = Tool.pearson_sim(o_b, bi, b_u)
                if val > 0.0: vals.append((val, o_b, r))

            if not vals: return fb

            vals.sort(key=lambda k: -k[0])
            top  = vals[:Tool.N]
            m, n = 0, 0
            for s, o_b, r in top:
                l  = s * abs(s)
                m += l * (r - avg_b_r.get(o_b, g_avg))
                n += abs(s)

            if n == 0: return fb
            return max(1.0, min(5.0, avg_b_r[bi] + (m / n)))

        def feat_eng(u_i, b_i):
            ft_u = feat_u.get(u_i, {})
            ft_b = feat_b.get(b_i, {})
            return [
                ft_b.get("b_json_avg", g_avg),
                ft_u.get("u_json_avg", g_avg),
                ft_b.get("b_lat", 0.0),
                ft_u.get("u_rv_c", 0),
                ft_b.get("b_price", 0),
                t_b.get(b_i, 0),
                ft_u.get("compliments", 0),
                ft_u.get("useful", 0),
                ft_b.get("b_long", 0.0),
                ft_b.get("b_cat", 0),
                feat_ch.get(b_i, 0),
                ft_b.get("b_rv_c", 0),
                t_u.get(u_i, 0),
            ]

        my_tr_d = trrdd.collect()
        X_train  = [feat_eng(u_i, b_i) for u_i, b_i, _ in my_tr_d]
        y_train  = [st for _, _, st in my_tr_d]

        model = XGBRegressor(
            n_estimators=700,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=553,
            tree_method='hist'
        )
        model.fit(np_arr(X_train), np_arr(y_train))

        my_tt_d = ttrdd.collect()
        X_test  = [feat_eng(u_i, b_i) for u_i, b_i in my_tt_d]
        res     = model.predict(np_arr(X_test))

        write_data = ['user_id, business_id, prediction']
        for m, (u_i, b_i) in enumerate(my_tt_d):
            x_pr = float(res[m])
            if u_i in u_b and b_i in b_u:
                nn = len([1 for o_b in u_b[u_i] if o_b != b_i and Tool.pearson_sim(o_b, b_i, b_u) > 0.0])
            else:
                nn = 0
            if nn >= 15:  a = 0.3
            elif nn >= 5: a = 0.15
            else:         a = 0.05
            pr = pr_rat(u_i, b_i)
            j  = a * pr + (1 - a) * x_pr
            j  = max(1.0, min(5.0, j))
            write_data.append(f'{u_i},{b_i},{j}')

        Tool.write_output(write_data, out_path)

        # error distribution
        preds = {}
        with open(out_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                preds[(row[0].strip(), row[1].strip())] = float(row[2])

        true = {}
        with open(test_fp, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) == 3:
                    true[(row[0].strip(), row[1].strip())] = float(row[2])

        if true:
            diffs = [abs(preds[k] - true[k]) for k in true if k in preds]
            rmse  = math.sqrt(sum((preds[k] - true[k])**2 for k in true if k in preds) / len(diffs))
            print(f">=0 and <1 : {sum(1 for d in diffs if 0 <= d < 1)}")
            print(f">=1 and <2 : {sum(1 for d in diffs if 1 <= d < 2)}")
            print(f">=2 and <3 : {sum(1 for d in diffs if 2 <= d < 3)}")
            print(f">=3 and <4 : {sum(1 for d in diffs if 3 <= d < 4)}")
            print(f">=4        : {sum(1 for d in diffs if d >= 4)}")
            print(f"RMSE: {rmse:.6f}")

        print(f"Done in {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: competition.py <folder_path> <test_file> <output_file>')
    else:
        s = Solution()
        s.start()