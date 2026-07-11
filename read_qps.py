from collections import OrderedDict
import cvxpy as cp
import re
import numpy as np
import scipy.sparse as sps


def read_qps(f):
    rows = OrderedDict()
    rhs = OrderedDict()
    columns = OrderedDict()
    Adata = []
    Arow = []
    Acol = []
    ranges = OrderedDict()
    bounds = OrderedDict()
    Pdata = []
    Prow = []
    Pcol = []
    header = None
    for line in f:
        line = line.rstrip()
        if line == "":
            continue
        if header is not None:
            if line[0] == " ":
                entries = re.split(r"\s+", line.lstrip())
                entries4 = re.split(r"\s+", line[4:])
                if header == "NAME":
                    pass
                elif len(entries) == 0:
                    pass
                elif header == "ROWS":
                    if entries[1] in rows:
                        raise ValueError(f"Duplicate row '{line[4:]}'")
                    else:
                        rows[entries[1]] = (len(rows), entries[0])
                elif header == "COLUMNS":
                    if entries[0] not in columns:
                        columns[entries[0]] = len(columns)
                    if len(entries) % 2 != 1:
                        raise ValueError(f"Invalid column '{line}'")
                    for i in range(1, len(entries), 2):
                        Adata.append(float(entries[i + 1]))
                        Arow.append(entries[i])
                        Acol.append(entries[0])
                elif header == "RHS":
                    entries = entries4
                    if entries[0] not in rhs:
                        rhs[entries[0]] = []
                    if len(entries) % 2 != 1:
                        raise ValueError(f"Invalid rhs '{line}'")
                    for i in range(1, len(entries), 2):
                        val = float(entries[i + 1])
                        rhs[entries[0]].append((entries[i], val))
                elif header == "RANGES":
                    # why does this exist
                    if entries[0] not in ranges:
                        ranges[entries[0]] = []
                    if len(entries) % 2 != 1:
                        raise ValueError(f"Invalid ranges '{line}'")
                    for i in range(1, len(entries), 2):
                        val = float(entries[i + 1])
                        ranges[entries[0]].append((entries[i], val))
                elif header == "BOUNDS":
                    entries = [entries[0]] + entries4
                    typ = entries[0]
                    if entries[1] not in bounds:
                        bounds[entries[1]] = []
                    if len(entries) != 4 and len(entries) != 3:
                        raise ValueError(f"Invalid bounds '{line}'")
                    val = float(entries[3]) if len(entries) == 4 else None
                    bounds[entries[1]].append((typ, entries[2], val))
                elif header == "QUADOBJ":
                    if len(entries) != 3:
                        raise ValueError(f"Invalid quadobj '{line}'")
                    val = float(entries[2])
                    Pdata.append(val)
                    Prow.append(entries[0])
                    Pcol.append(entries[1])
                    if entries[0] != entries[1]:
                        Pdata.append(val)
                        Prow.append(entries[1])
                        Pcol.append(entries[0])
                else:
                    raise ValueError(f"Invalid header '{header}'")
            else:
                header = None
        if header is None:
            if line[0] == " ":
                raise ValueError("file starts with indented line")
            header = line.split(" ")[0]
            if header == "ENDATA":
                break

    if len(rhs) > 1:
        raise ValueError("More than 1 RHS vector")
    if len(ranges) > 1:
        raise ValueError("More than 1 RANGES vector")
    if len(bounds) > 1:
        raise ValueError("More than 1 BOUNDS vector")

    objrow = None
    for row in rows:
        if rows[row][1] == "N":
            if objrow is None:
                objrow = row
            else:
                raise ValueError("More than one N row")
    if objrow is None:
        raise ValueError("No N row")
    objind = rows[objrow][0]

    Arowi = []
    for row in Arow:
        if row not in rows:
            raise ValueError(f"Invalid row '{row}' in COLUMNS")
        Arowi.append(rows[row][0])
    Acoli = [columns[col] for col in Acol]
    Ashape = (len(rows), len(columns))
    A = sps.coo_array((Adata, (Arowi, Acoli)), shape=Ashape)

    rhs = [] if len(rhs) == 0 else list(rhs.values())[0]
    rhsv = np.full(len(rows), np.nan)
    for p in rhs:
        if p[0] not in rows:
            raise ValueError(f"Invalid row '{p[0]}' in RHS")
        if not np.isnan(rhsv[rows[p[0]][0]]):
            raise ValueError(f"Double entry for row '{p[0]}' in RHS")
        rhsv[rows[p[0]][0]] = p[1]
    rhsv[np.isnan(rhsv)] = 0

    lranges = [] if len(ranges) == 0 else list(ranges.values())[0]
    ranges = {}
    for row, val in lranges:
        if row in ranges:
            raise ValueError(f"Duplicate row '{row}' in RANGES")
        ranges[row] = val

    bounds = [] if len(bounds) == 0 else list(bounds.values())[0]
    lobound = np.full(len(columns), np.nan)
    upbound = np.full(len(columns), np.inf)
    for typ, col, val in bounds:
        if col not in columns:
            raise ValueError(f"Invalid column '{col}' in BOUNDS")
        ind = columns[col]
        if typ == "LO":
            if val is None:
                raise ValueError(f"Missing value for BOUND on '{col}'")
            lobound[ind] = val
        elif typ == "UP":
            if val is None:
                raise ValueError(f"Missing value for BOUND on '{col}'")
            upbound[ind] = val
        elif typ == "FX":
            if val is None:
                raise ValueError(f"Missing value for BOUND on '{col}'")
            lobound[ind] = val
            upbound[ind] = val
        elif typ == "FR":
            lobound[ind] = -np.inf
            upbound[ind] = np.inf
        elif typ == "MI":
            lobound[ind] = -np.inf
        elif typ == "PL":
            upbound[ind] = np.inf
        else:
            raise ValueError(f"Invalid BOUND type '{typ}'")
    for ind in range(len(columns)):
        if np.isnan(lobound[ind]):
            if upbound[ind] <= 0:
                lobound[ind] = -np.inf
            else:
                lobound[ind] = 0
        elif upbound[ind] < lobound[ind]:
            raise ValueError(f"Contradictory bounds for '{col}'")

    Prowi = []
    for row in Prow:
        if row not in columns:
            raise ValueError(f"Invalid name '{row}' in QUADOBJ")
        Prowi.append(columns[row])
    Pcoli = [columns[col] for col in Pcol]
    Pshape = (len(columns), len(columns))
    P = sps.coo_array((Pdata, (Prowi, Pcoli)), shape=Pshape)

    X = cp.Variable(len(columns))
    Y = A @ X
    obj = (1 / 2) * cp.quad_form(X, cp.psd_wrap(P)) + Y[objind]

    constraints = []
    for row in rows:
        val = None
        if row in ranges:
            val = ranges[row]
        ind, typ = rows[row]
        if typ == "N":
            continue
        if typ == "E" or val == 0:
            if val is None or val == 0:
                constraints.append(Y[ind] == rhsv[ind])
            elif val < 0:
                constraints.append(Y[ind] <= rhsv[ind])
                constraints.append(Y[ind] >= rhsv[ind] + val)
            else:
                constraints.append(Y[ind] >= rhsv[ind])
                constraints.append(Y[ind] <= rhsv[ind] + val)
        elif typ == "L":
            constraints.append(Y[ind] <= rhsv[ind])
            if val is not None:
                constraints.append(Y[ind] >= rhsv[ind] - np.abs(val))
        elif typ == "G":
            constraints.append(Y[ind] >= rhsv[ind])
            if val is not None:
                constraints.append(Y[ind] <= rhsv[ind] + np.abs(val))
        else:
            raise ValueError(f"Invalid row type '{typ}'")

    for ind in range(len(columns)):
        if lobound[ind] != -np.inf:
            constraints.append(X[ind] >= lobound[ind])
        if upbound[ind] != np.inf:
            constraints.append(X[ind] <= upbound[ind])

    return obj, constraints, X
