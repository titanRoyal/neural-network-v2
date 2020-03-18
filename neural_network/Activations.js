class Activations {
    constructor() {
        Activations.sigmoid = new active("sigmoid", (x) => {
            return 1 / (1 + Math.exp(-x))
        }, (x) => {
            return x * (1 - x);
        });

        Activations.tanH = new active("tanH", (x) => {
            return (2 / (1 + Math.exp(-2 * x))) - 1;
        }, (x) => {
            let r = sigmoid(x)
            return 1 - r * r;
        });

        Activations.identity = new active("identity", (x) => {
            return x;
        }, (x) => {
            return 1;
        });
        Activations.softSign = new active("softSign", (x) => {
            return Math.atan(x);
        }, (x) => {
            return 1 / (Math.pow(x, 2) + 1)
        });
        Activations.relu = new active("relu", (x) => {
            if (x < 0) {
                return 0
            } else {
                return x;
            }
        }, (x) => {
            if (x < 0) {
                return 0;
            } else {
                return 1;
            }
        });

    }

}
class active {
    constructor(type, func, dFunc) {
        this.type = type;
        this.func = func;
        this.dFunc = dFunc;
    }
    forward(x) {
        return this.func(x);
    }
    backward(x) {
        return this.dFunc(x);
    }
}