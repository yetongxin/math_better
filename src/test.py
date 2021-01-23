def get_all_eq(prefix_seq, ops, ex_ops):
    st = []
    for i in range(len(prefix_seq) - 1, -1, -1):
        t = prefix_seq[i]
        if t not in ops:
            st.append([[t]])
        elif t in ex_ops:
            a = st.pop()
            b = st.pop()
            temp = []
            for x in a:
                for y in b:
                    temp.append([t] + x + y)
                    temp.append([t] + y + x)
            st.append(temp)
        else:
            a = st.pop()
            b = st.pop()
            temp = []
            for x in a:
                for y in b:
                    temp.append([t] + x + y)
            st.append(temp)
        print('st:', st)
    return st[0]

ops = ['+', '-', '*', '/', '^']
ex_ops = ['+', '*']
# seq = ['+', '*', 'N0', '+', '1', 'N1', 'N0']
seq = ['-', 'N0', '+', 'N1', 'N2']
res = get_all_eq(seq, ops, ex_ops)


print(res)