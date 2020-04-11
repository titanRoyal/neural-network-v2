function neural_net(inp, nh, o, options = {}) {
  new Activations();
  this.initParams = function (options) {
    if (!isNaN(options["learningRate"]) && options["learningRate"] <= 1 && options["learningRate"] > 0) {
      this.lr = options["learningRate"];
    } else {
      this.lr = 0.05;
    }
    if (!isNaN(options["mutateRate"]) && options["mutateRate"] <= 1 && options["mutateRate"] > 0) {
      this.mr = options["muateRate"];
    } else {
      this.mr = 0.3;
    }
    if (options["useMomentum"]) {
      if (options["useMomentum"] == true) {
        this.momentCount = 4;
      } else if (!isNaN(options["useMomentum"])) {
        this.momentCount = options["useMomentum"]
      } else {
        this.momentCount = 1
      }
    } else {
      this.momentCount = 1;
    }
    if (options["initActivation"]) {
      this.Activation = [options["initActivation"]]
    } else {
      this.Activation = ["sigmoid"]
    }
  }
  if (inp instanceof neural_net) {
    this.input_n = private(inp.input_n);
    this.hiden_n = private(inp.hiden_n);
    this.hiden_layer = private(inp.hiden_layer);
    this.output_n = private(inp.output_n);
    this.lr = private(inp.lr);
    this.weight = private([]);
    this.bias = private([]);
    // this.Activation=loadJSON("./Activations.json")||[];
    for (var i = 0; i < inp.weight.length; i++) {
      this.weight[i] = inp.weight[i].copy()
    }
    for (i = 0; i < inp.bias.length; i++) {
      this.bias[i] = inp.bias[i].copy()
    }
    this.Activation=[...inp.Activation];
    this.momentCount=inp.momentCount;
    this.mr=inp.mr;
  } else {
    this.input_n = inp;
    this.hiden_n = nh.map(data => {
      if (data instanceof Array) {
        return data[0]
      }
      return data;
    });
    this.hiden_layer = nh.length;
    this.output_n = o;
    this.weight = [];
    this.bias = []
    this.initParams(options);
    this.acc = []
    nh.forEach(data => {
      if (data instanceof Array) {

        if (data.length == 2) {
          this.Activation.push(data[1])
        } else {
          this.Activation.push("sigmoid")
        }
      } else {
        this.Activation.push("sigmoid")
      }
    })
    for (var i = 0; i < this.hiden_layer + 1; i++) {
      if (i == 0) {
        this.weight[i] = new Matrix(this.hiden_n[i], this.input_n);
      } else if (i == this.hiden_layer) {
        this.weight[i] = new Matrix(this.output_n, this.hiden_n[i - 1]);
        this.bias[i] = new Matrix(this.output_n, 1);
        this.bias[i].randomize();
      } else {
        this.weight[i] = new Matrix(this.hiden_n[i], this.hiden_n[i - 1]);
        this.bias[i] = new Matrix(this.hiden_n[i], 1);
        this.bias[i].randomize();
      }
      this.weight[i].randomize();


    }
    for (var i = 0; i < this.hiden_layer + 1; i++) {
      if (i == this.hiden_layer) {
        this.bias[i] = new Matrix(this.output_n, 1);
      } else {
        this.bias[i] = new Matrix(this.hiden_n[i], 1);
      }
      this.bias[i].randomize();
    }
  }

  this.feedforward = function (inp) {
    let inputs = Matrix.fromarray(inp);
    let sum_h;
    for (var i = 0; i < this.weight.length; i++) {
      sum_h = Matrix.mult(this.weight[i], inputs);
      sum_h.add(this.bias[i]);
      sum_h.map(Activations[this.Activation[i]].func);
      inputs = sum_h;
    }
    return Matrix.toarray(inputs);
  }
  this.train = function (input, answer, func) {
    if (input[0] instanceof Array) {
      try {
        if (answer[0] instanceof Array && input.length == answer.length) {
          for (let index = 0; index < input.length; index++) {
            let r = this.train(input[index], answer[index]);
            if (!r.success) {
              if (func) {
                func({
                  msg: "the input data and the output data must match the input layer also the output layer test set " + (index + 1),
                  err: {
                    input_Layer: this.input_n,
                    current_input: input[index].length,
                    output_Layer: this.output_n,
                    current_output: answer[index].length
                  }
                }, undefined)
              } else {
                return {
                  msg: "the input data and the output data must match the input layer also the output layer test set " + (index + 1),
                  err: {
                    input_Layer: this.input_n,
                    current_input: input[index].length,
                    output_Layer: this.output_n,
                    current_output: answer[index].length
                  }
                }
              }
              return;
            }
          }
          if (func) {
            func(undefined, {
              msg: "everything is good",
            })
          } else {
            return {
              msg: "everything is good",
            }
          }
          return;
        }
      } catch (error) {
        if (func) {
          func({
            msg: "the input data and the output data must match the input layer also the output layer",
            err: error
          }, undefined)
        } else {
          return {
            msg: "the input data and the output data must match the input layer also the output layer",
            err: error
          }
        }
        return;
      }
    }
    if ((!input instanceof Array) || (!answer instanceof Array)) {
      if (func) {
        func({
          msg: "make sure the first and the second argument are arrays",
          success: false,
        }, undefined)
        return;
      } else {
        return {
          msg: "make sure the first and the second argument are arrays",
          success: false
        }
      }
    } else if (input.length != this.input_n || answer.length != this.output_n) {
      if (func) {
        func({
          msg: "make sure the length of the arrays matches the length of the inputs",
          success: false,
          err: {
            input_Layer: this.input_n,
            current_input: input.length,
            output_Layer: this.output_n,
            current_output: answer.length
          }
        }, undefined)
        return;
      } else {
        return {
          msg: "make sure the length of the arrays matches the length of the inputs",
          success: false,
          err: {
            input_Layer: this.input_n,
            current_input: input.length,
            output_Layer: this.output_n,
            current_output: answer.length
          }
        }
      }
    }
    try {
      let inputs = Matrix.fromarray(input);
      let target = Matrix.fromarray(answer);
      let output;
      let err_tab = [];
      let gradiant = [];
      let deltaw = [];

      let sum_h = [];
      sum_h[0] = inputs;
      for (var i = 0; i < this.weight.length; i++) {
        sum_h[i] = Matrix.mult(this.weight[i], sum_h[i]);
        sum_h[i].add(this.bias[i]);
        sum_h[i].map(Activations[this.Activation[i]].func);
        sum_h[i + 1] = sum_h[i];
      }
      sum_h.splice(this.weight.length, 1);
      output = sum_h[sum_h.length - 1];
      let sco = 0;
      Matrix.substract(target, output).matrox.forEach(data => {
        data.forEach(d => {
          sco += Math.abs(d);
        })
      })
      this.lr = sco / this.output_n;


      let output_err = Matrix.substract(target, output);
      for (var i = this.weight.length - 1; i >= 0; i--) {
        if (i == this.weight.length - 1) {
          err_tab[i] = Matrix.mult(Matrix.transpose(this.weight[i]), output_err);
        } else {
          err_tab[i] = Matrix.mult(Matrix.transpose(this.weight[i]), err_tab[i + 1])
        }
      }
      for (var i = 0; i < this.weight.length; i++) {
        gradiant[i] = Matrix.map(sum_h[i], Activations[this.Activation[i]].dFunc);
        if (i == 0) {
          gradiant[i].mult(err_tab[i + 1]);
          gradiant[i].mult(this.lr);
          if (!this.acc[i]) {
            this.acc[i] = new Momentum(this.momentCount);
          }
          this.acc[i].add(gradiant[i]);
          gradiant[i] = this.acc[i].calculate();
          this.bias[i].add(gradiant[i]);
          let inppp_t = Matrix.transpose(inputs);
          deltaw[i] = Matrix.mult(gradiant[i], inppp_t);

        } else if (i == this.weight.length - 1) {
          gradiant[i].mult(output_err);
          gradiant[i].mult(this.lr);
          if (!this.acc[i]) {
            this.acc[i] = new Momentum(this.momentCount);
          }
          this.acc[i].add(gradiant[i]);
          gradiant[i] = this.acc[i].calculate();
          this.bias[i].add(gradiant[i]);
          deltaw[i] = Matrix.mult(gradiant[i], Matrix.transpose(sum_h[i - 1]));
        } else {
          gradiant[i].mult(err_tab[i + 1]);
          gradiant[i].mult(this.lr);
          if (!this.acc[i]) {
            this.acc[i] = new Momentum(this.momentCount);
          }
          this.acc[i].add(gradiant[i]);
          gradiant[i] = this.acc[i].calculate();
          this.bias[i].add(gradiant[i]);
          deltaw[i] = Matrix.mult(gradiant[i], Matrix.transpose(sum_h[i - 1]));
        }
        this.weight[i].add(deltaw[i]);
      }
      // FIXME: more information can be added to the data returned by the training function
      if (func) {
        func(undefined, this.msg("everything is good", true))
      } else {
        return this.msg("everything is good", true)
      }
    } catch (error) {
      if (func) {
        func(this.msg("the input data and the output data must match the input layer also the output layer", false, error), undefined)
      } else {
        return this.msg("the input data and the output data must match the input layer also the output layer", false)
      }
    }
  }
  this.msg = function (m, s, err = undefined) {
    return {
      msg: m,
      success: s,
      err: err
    }
  }
  this.clone = function (download = false) {
    let t = {
      weights: [],
      biases: [],
      Activations: [...this.Activation]
    }
    this.weight.forEach((data) => {
      t.weights.push(data.copy())
    })
    this.bias.forEach((data) => {
      t.biases.push(data.copy())
    })
    if (download) {
      saveJSON(t, "Model.json")
    } else {
      return t;
    }
  }

  this.upload = function (obj) {
    if (obj.weights.length != this.weight.length || obj.biases.length != this.bias.length) {
      console.log("data structure passed is incompatible");
      return;
    }
    let test = true;
    obj.weights.forEach((data, index) => {
      if (data.row != this.weight[index].row || data.col != this.weight[index].col) {
        test = false;
        return;
      };
    })
    obj.biases.forEach((data, index) => {
      if (test && data.row != this.bias[index].row || data.col != this.bias[index].col) {
        test = false;
        return;
      };
    })
    obj.Activations.forEach((data, index) => {
      if (test && data != this.Activation[index]) {
        test = false;
        return;
      }
    })
    if (!test) {
      console.log("data structure passed is incompatible");
      return;
    }
    obj.weights.forEach((data, index) => {
      this.weight[index] = Matrix.copy(data);
    })
    obj.biases.forEach((data, index) => {
      this.bias[index] = Matrix.copy(data);
    })
    this.Activation = [...obj.Activations]
    console.log("data loaded");


  }
  this.copy = function () {
    return new neural_net(this);
  }
  this.mutate = function (func) {
    for (var i = 0; i < this.weight.length; i++) {
      this.weight[i].map(func)
    }
    for (var i = 0; i < this.bias.length; i++) {
      this.bias[i].map(func)
    }

  }
}

class Momentum {
  constructor(len = 4, moment = .9) {
    this.momentum = moment;
    this.max = len;
    this.tab = []
  }
  add(gradiant) {
    if (this.tab.length >= this.max) {
      this.tab.shift()
    }
    this.tab.unshift(gradiant)
  }
  calculate() {
    let mat = new Matrix(this.tab[0].row, this.tab[0].col)
    this.tab.forEach((data, index) => {
      let i = Math.pow(this.momentum, index);
      mat.add(Matrix.mult(data, i))
    })
    mat.mult((1 / this.tab.length));
    return mat;
  }
}