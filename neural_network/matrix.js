class Matrix {
  constructor(x, y) {
    this.row = x;
    this.col = y;
    this.matrox = []
    for (let i = 0; i < this.row; i++) {
      this.matrox[i] = []
      for (let j = 0; j < this.col; j++) {
        this.matrox[i][j] = 0;
      }
    }
  }
  copy() {
    let m = new Matrix(this.row, this.col);
    for (var i = 0; i < this.row; i++) {
      for (var j = 0; j < this.col; j++) {
        m.matrox[i][j] = this.matrox[i][j];
      }
    }
    return m;
  }
  static copy(obj) {
    let m = new Matrix(obj.row, obj.col);
    for (var i = 0; i < obj.row; i++) {
      for (var j = 0; j < obj.col; j++) {
        m.matrox[i][j] = obj.matrox[i][j];
      }
    }
    return m;
  }
  static crossover(a, b) {
    let m = new Matrix(a.row, a.col)
    for (var i = 0; i < a.row; i++) {
      let h = random(a.col);
      for (var j = 0; j < a.col; j++) {
        if (j > h) {
          m.matrox[i][j] = a.matrox[i][j];
        } else {
          m.matrox[i][j] = b.matrox[i][j];

        }
      }
    }
    return m;
  }
  static percentage(xy) {
    if (xy instanceof Matrix) {
      let r = 0;
      for (var i = 0; i < xy.row; i++) {
        for (var j = 0; j < xy.col; j++) {
          r += xy.matrox[i][j];
        }
      }
      let f = new Matrix(xy.row, xy.col);
      for (var i = 0; i < xy.row; i++) {
        for (var j = 0; j < xy.col; j++) {
          f.matrox[i][j] = xy.matrox[i][j] / r;;
        }
      }
      return f;
    } else if (xy instanceof Array) {
      let r = 0;
      for (var i = 0; i < xy.length; i++) {
        r += xy[i];
      }
      let f = [];
      for (var i = 0; i < xy.length; i++) {
        f.push(xy[i] /= r);
      }
      return f;
    }
  }
  static addall(xy) {
    let r = 0;
    for (var i = 0; i < xy.row; i++) {
      for (var j = 0; j < xy.col; j++) {
        r += xy.matrox[i][j];
      }
    }
    return r;
  }
  static normo(xc, g) {
    if (g == "p" || g == "P") {
      let jj = Matrix.addall(xc)
      let new_ho = new Matrix(xc.row, xc.col);
      for (var i = 0; i < xc.row; i++) {
        for (var j = 0; j < xc.col; j++) {
          new_ho.matrox[i][j] = xc.matrox[i][j] * 100 / jj;
        }
      }
      return new_ho;
    } else if (g == "n" || g == "N") {
      let jj = Matrix.addall(xc)
      let new_ho = new Matrix(xc.row, xc.col);
      for (var i = 0; i < xc.row; i++) {
        for (var j = 0; j < xc.col; j++) {
          new_ho.matrox[i][j] = xc.matrox[i][j] / jj;
        }
      }
      return new_ho;
    }

  }
  static substract(a, b) {
    let result = new Matrix(a.row, a.col);
    for (var i = 0; i < a.row; i++) {
      for (var j = 0; j < a.col; j++) {
        result.matrox[i][j] = a.matrox[i][j] - b.matrox[i][j]
      }
    }
    return result;
  }
  static mult(mi, n) {
    if (n instanceof Matrix) {
      if (mi.col == n.row) {
        let m = new Matrix(mi.row, n.col);
        for (let i = 0; i < m.row; i++) {
          for (let j = 0; j < m.col; j++) {
            let sum = 0;
            let count = 0;
            while (count < mi.col) {
              sum += mi.matrox[i][count] * n.matrox[count][j]
              count++
            }
            m.matrox[i][j] = sum;
          }
        }
        return m;
      } else {
        console.log("Matrix dont match");
      }
    } else {
      let m = mi.copy();
      mi.mult(n)
      return m;

    }
  }
  static transpose(klm) {
    let r = new Matrix(klm.col, klm.row)
    for (let i = 0; i < klm.row; i++) {
      for (let j = 0; j < klm.col; j++) {
        r.matrox[j][i] = klm.matrox[i][j];
      }
    }
    return r;
  }

  static oneline(func) {
    let s = new Matrix(0, 0)
    for (let i = 0; i < func.row; i++) {
      for (let j = 0; j < func.col; j++) {
        s.matrox.push(func.matrox[i][j]);
      }
    }
    return s;
  }

  static toarray(func) {
    let s = []
    for (let i = 0; i < func.row; i++) {
      for (let j = 0; j < func.col; j++) {
        s.push(func.matrox[i][j]);
      }
    }
    return s;
  }
  static fromarray(arr) {
    let m = new Matrix(arr.length, 1);
    for (var i = 0; i < arr.length; i++) {
      m.matrox[i][0] = arr[i];
    }
    return m;
  }
  static map(a, func) {
    let m = new Matrix(a.row, a.col)
    for (let i = 0; i < a.row; i++) {
      for (let j = 0; j < a.col; j++) {
        m.matrox[i][j] = func(a.matrox[i][j]);
      }
    }
    return m;
  }
  static mapp(a, func) {
    let m = new Matrix(a.row, a.col)
    for (let i = 0; i < a.row; i++) {
      for (let j = 0; j < a.col; j++) {
        m.matrox[i][j] = func(a);
      }
    }
    return m;
  }

}
Matrix.prototype.add = function (n) {
  if (n instanceof Matrix) {
    for (let i = 0; i < this.row; i++) {
      for (let j = 0; j < this.col; j++) {
        this.matrox[i][j] += n.matrox[i][j];
      }
    }
  } else {
    for (let i = 0; i < this.row; i++) {
      for (let j = 0; j < this.col; j++) {
        this.matrox[i][j] += n;
      }
    }
  }
}
Matrix.prototype.mult = function (n) {
  if (n instanceof Matrix) {
    for (let i = 0; i < this.row; i++) {
      for (let j = 0; j < this.col; j++) {
        this.matrox[i][j] *= n.matrox[i][j];
      }
    }
  } else {
    for (let i = 0; i < this.row; i++) {
      for (let j = 0; j < this.col; j++) {
        this.matrox[i][j] *= n;
      }
    }
  }
}
Matrix.prototype.print = function () {
  console.table(this.matrox);
}
Matrix.prototype.randomize = function () {
  for (let i = 0; i < this.row; i++) {
    for (let j = 0; j < this.col; j++) {
      this.matrox[i][j] = Math.random() * 2 - 1;
    }
  }

}
Matrix.prototype.transpose = function () {
  let r = new Matrix(this.col, this.row)
  for (let i = 0; i < this.row; i++) {
    for (let j = 0; j < this.col; j++) {
      r.matrox[j][i] = this.matrox[i][j];
    }
  }
  this.matrox = r.matrox;
}
Matrix.prototype.map = function (func) {
  for (let i = 0; i < this.row; i++) {
    for (let j = 0; j < this.col; j++) {
      this.matrox[i][j] = func(this.matrox[i][j]);
    }
  }
}