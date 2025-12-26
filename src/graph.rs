use std::collections::HashMap;
use crate::ast::{BinaryOperator, Expression, UnaryOperator};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new(id: usize) -> Self {
        NodeId(id)
    }

    pub fn index(&self) -> usize {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if expected != data.len() {
            return Err(format!("Tensor data/shape mismatch: expected {}, got {}", expected, data.len()));
        }
        Ok(Tensor { data, shape })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Tensor { data: vec![0.0; size], shape }
    }

    pub fn zeros_like(&self) -> Self {
        Tensor::zeros(self.shape.clone())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(f64),
    Tensor(Tensor),
}

impl Value {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_tensor(&self) -> Option<&Tensor> {
        match self {
            Value::Tensor(t) => Some(t),
            _ => None,
        }
    }

    pub fn zeros_like(&self) -> Value {
        match self {
            Value::Scalar(_) => Value::Scalar(0.0),
            Value::Tensor(t) => Value::Tensor(t.zeros_like()),
        }
    }

    pub fn ones_like(&self) -> Value {
        match self {
            Value::Scalar(_) => Value::Scalar(1.0),
            Value::Tensor(t) => Value::Tensor(Tensor { data: vec![1.0; t.data.len()], shape: t.shape.clone() }),
        }
    }

    pub fn map_unary<F>(&self, f: F) -> Result<Value, String>
    where
        F: Fn(f64) -> f64,
    {
        match self {
            Value::Scalar(v) => Ok(Value::Scalar(f(*v))),
            Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| f(*x)).collect(), shape: t.shape.clone() })),
        }
    }

    pub fn map2<F>(&self, other: &Value, f: F) -> Result<Value, String>
    where
        F: Fn(f64, f64) -> f64,
    {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(f(*a, *b))),
            (Value::Tensor(t1), Value::Tensor(t2)) => {
                if t1.shape != t2.shape {
                    return Err("Tensor shape mismatch".to_string());
                }
                let data = t1.data.iter().zip(t2.data.iter()).map(|(a, b)| f(*a, *b)).collect();
                Ok(Value::Tensor(Tensor { data, shape: t1.shape.clone() }))
            }
            (Value::Scalar(s), Value::Tensor(t)) => {
                let data = t.data.iter().map(|b| f(*s, *b)).collect();
                Ok(Value::Tensor(Tensor { data, shape: t.shape.clone() }))
            }
            (Value::Tensor(t), Value::Scalar(s)) => {
                let data = t.data.iter().map(|a| f(*a, *s)).collect();
                Ok(Value::Tensor(Tensor { data, shape: t.shape.clone() }))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>,
    pub value: Option<Value>,
    pub gradient: Option<Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Constant(Value),
    Learnable(String),
    Variable(String),
    BinaryOp(String),
    UnaryOp(String),
    FunctionCall(String),
}

#[derive(Debug, Clone)]
pub struct ComputationalGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: usize,
    learnables: Vec<String>,
}

impl ComputationalGraph {
    pub fn new() -> Self {
        ComputationalGraph {
            nodes: HashMap::new(),
            next_id: 0,
            learnables: Vec::new(),
        }
    }

    pub fn add_constant(&mut self, value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Constant(Value::Scalar(value)),
            inputs: Vec::new(),
            value: Some(Value::Scalar(value)),
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_constant_tensor(&mut self, data: Vec<f64>, shape: Vec<usize>) -> Result<NodeId, String> {
        let tensor = Tensor::new(data, shape)?;
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Constant(Value::Tensor(tensor.clone())),
            inputs: Vec::new(),
            value: Some(Value::Tensor(tensor)),
            gradient: None,
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn add_learnable(&mut self, name: String, initial_value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.learnables.push(name.clone());

        let node = Node {
            id,
            node_type: NodeType::Learnable(name),
            inputs: Vec::new(),
            value: Some(Value::Scalar(initial_value)),
            gradient: Some(Value::Scalar(0.0)),
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_learnable_tensor(&mut self, name: String, data: Vec<f64>, shape: Vec<usize>) -> Result<NodeId, String> {
        let tensor = Tensor::new(data, shape)?;
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.learnables.push(name.clone());

        let grad = Value::Tensor(Tensor::zeros(tensor.shape.clone()));

        let node = Node {
            id,
            node_type: NodeType::Learnable(name),
            inputs: Vec::new(),
            value: Some(Value::Tensor(tensor)),
            gradient: Some(grad),
        };

        self.nodes.insert(id, node);
        Ok(id)
    }

    pub fn add_variable(&mut self, name: String, input: NodeId) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Variable(name),
            inputs: vec![input],
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_binary_op(&mut self, op: &str, left: NodeId, right: NodeId) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::BinaryOp(op.to_string()),
            inputs: vec![left, right],
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_unary_op(&mut self, op: &str, operand: NodeId) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::UnaryOp(op.to_string()),
            inputs: vec![operand],
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn add_function_call(&mut self, name: String, args: Vec<NodeId>) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::FunctionCall(name),
            inputs: args,
            value: None,
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    pub fn build_from_expression(&mut self, expr: &Expression, variables: &HashMap<String, NodeId>) -> Result<NodeId, String> {
        match expr {
            Expression::Number(n) => Ok(self.add_constant(*n)),
            Expression::Identifier(name) => variables.get(name).copied().ok_or_else(|| format!("Undefined variable: {}", name)),
            Expression::BinaryOp { left, op, right } => {
                let left_id = self.build_from_expression(left, variables)?;
                let right_id = self.build_from_expression(right, variables)?;

                let op_str = match op {
                    BinaryOperator::Add => "add",
                    BinaryOperator::Sub => "sub",
                    BinaryOperator::Mul => "mul",
                    BinaryOperator::Div => "div",
                    BinaryOperator::Mod => "mod",
                    BinaryOperator::Pow => "pow",
                    BinaryOperator::Equal => "eq",
                    BinaryOperator::NotEqual => "ne",
                    BinaryOperator::Less => "lt",
                    BinaryOperator::Greater => "gt",
                    BinaryOperator::LessEq => "le",
                    BinaryOperator::GreaterEq => "ge",
                    BinaryOperator::And => "and",
                    BinaryOperator::Or => "or",
                };

                Ok(self.add_binary_op(op_str, left_id, right_id))
            }
            Expression::UnaryOp { op, expr } => {
                let expr_id = self.build_from_expression(expr, variables)?;
                let op_str = match op {
                    UnaryOperator::Neg => "neg",
                    UnaryOperator::Not => "not",
                };
                Ok(self.add_unary_op(op_str, expr_id))
            }
            Expression::Call { name, args } => {
                let mut arg_ids = Vec::new();
                for arg in args {
                    arg_ids.push(self.build_from_expression(arg, variables)?);
                }
                Ok(self.add_function_call(name.clone(), arg_ids))
            }
        }
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    pub fn nodes(&self) -> &HashMap<NodeId, Node> {
        &self.nodes
    }

    pub fn learnables(&self) -> &[String] {
        &self.learnables
    }

    fn topological_order(&self) -> Result<Vec<NodeId>, String> {
        let mut indegree: HashMap<NodeId, usize> = HashMap::new();
        let mut adj: HashMap<NodeId, Vec<NodeId>> = HashMap::new();

        for (&id, node) in &self.nodes {
            indegree.insert(id, node.inputs.len());
            for input in &node.inputs {
                adj.entry(*input).or_default().push(id);
            }
        }

        let mut zero_indegree: Vec<NodeId> = indegree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        zero_indegree.sort_by_key(|n| n.index());

        let mut order = Vec::with_capacity(self.nodes.len());

        while let Some(id) = zero_indegree.first().cloned() {
            zero_indegree.remove(0);
            order.push(id);

            if let Some(children) = adj.get(&id) {
                for child in children {
                    if let Some(entry) = indegree.get_mut(child) {
                        *entry -= 1;
                        if *entry == 0 {
                            zero_indegree.push(*child);
                        }
                    }
                }
                zero_indegree.sort_by_key(|n| n.index());
            }
        }

        if order.len() != self.nodes.len() {
            return Err("Graph contains a cycle or disconnected nodes".to_string());
        }

        Ok(order)
    }

    pub fn forward_pass(&mut self) -> Result<(), String> {
        let node_ids = self.topological_order()?;

        for node_id in node_ids {
            let node_type = self.nodes.get(&node_id).map(|n| &n.node_type).cloned();
            let inputs = self.nodes.get(&node_id).map(|n| n.inputs.clone()).unwrap_or_default();

            if let Some(node_type) = node_type {
                match node_type {
                    NodeType::Constant(v) => {
                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.value = Some(v.clone());
                        }
                    }
                    NodeType::Learnable(_) => {}
                    NodeType::Variable(_) => {
                        if inputs.len() == 1 {
                            if let Some(input_val) = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()) {
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(input_val);
                                }
                            }
                        }
                    }
                    NodeType::BinaryOp(op) => {
                        if inputs.len() == 2 {
                            let left_val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing left operand")?;
                            let right_val = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone()).ok_or("Missing right operand")?;

                            let result = match op.as_str() {
                                "add" => left_val.map2(&right_val, |a, b| a + b)?,
                                "sub" => left_val.map2(&right_val, |a, b| a - b)?,
                                "mul" => left_val.map2(&right_val, |a, b| a * b)?,
                                "div" => left_val.map2(&right_val, |a, b| a / b)?,
                                "mod" => left_val.map2(&right_val, |a, b| a % b)?,
                                "pow" => left_val.map2(&right_val, |a, b| a.powf(b))?,
                                "eq" => Value::Scalar(if (left_val.as_scalar().unwrap_or(f64::NAN) - right_val.as_scalar().unwrap_or(f64::NAN)).abs() < f64::EPSILON { 1.0 } else { 0.0 }),
                                "ne" => Value::Scalar(if (left_val.as_scalar().unwrap_or(f64::NAN) - right_val.as_scalar().unwrap_or(f64::NAN)).abs() >= f64::EPSILON { 1.0 } else { 0.0 }),
                                "lt" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? < right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "gt" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? > right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "le" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? <= right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "ge" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? >= right_val.as_scalar().ok_or("Non-scalar right")? { 1.0 } else { 0.0 }),
                                "and" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? != 0.0 && right_val.as_scalar().ok_or("Non-scalar right")? != 0.0 { 1.0 } else { 0.0 }),
                                "or" => Value::Scalar(if left_val.as_scalar().ok_or("Non-scalar left")? != 0.0 || right_val.as_scalar().ok_or("Non-scalar right")? != 0.0 { 1.0 } else { 0.0 }),
                                _ => return Err(format!("Unknown binary op: {}", op)),
                            };

                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::UnaryOp(op) => {
                        if inputs.len() == 1 {
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing operand")?;

                            let result = match op.as_str() {
                                "neg" => val.map_unary(|v| -v)?,
                                "not" => {
                                    let s = val.as_scalar().ok_or("Non-scalar for not")?;
                                    Value::Scalar(if s != 0.0 { 0.0 } else { 1.0 })
                                }
                                _ => return Err(format!("Unknown unary op: {}", op)),
                            };

                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::FunctionCall(name) => match name.as_str() {
                        "sigmoid" => {
                            if inputs.len() == 1 {
                                let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                                let result = val.map_unary(|v| 1.0 / (1.0 + (-v).exp()))?;
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(result);
                                }
                            }
                        }
                        "relu" => {
                            if inputs.len() == 1 {
                                let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone()).ok_or("Missing argument")?;
                                let result = val.map_unary(|v| if v > 0.0 { v } else { 0.0 })?;
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(result);
                                }
                            }
                        }
                        _ => {}
                    },
                }
            }
        }
        Ok(())
    }

    pub fn print_structure(&self) {
        println!("=== Computational Graph ===");
        for (id, node) in &self.nodes {
            println!(
                "Node {:?}: {:?}, value: {:?}, gradients: {:?}, inputs: {:?}",
                id, node.node_type, node.value, node.gradient, node.inputs
            );
        }
    }

    pub fn backward_pass(&mut self, output_id: NodeId) -> Result<(), String> {
        if let Some(node) = self.nodes.get_mut(&output_id) {
            node.gradient = Some(node.value.clone().map(|v| v.ones_like()).unwrap_or(Value::Scalar(1.0)));
        }

        let mut node_ids = self.topological_order()?;
        node_ids.reverse();

        for node_id in node_ids {
            let grad_opt = self.get_node(node_id).and_then(|n| n.gradient.clone());
            let Some(gradient) = grad_opt else { continue };
            match &gradient {
                Value::Scalar(g) if *g == 0.0 => continue,
                Value::Tensor(t) if t.data.iter().all(|v| *v == 0.0) => continue,
                _ => {}
            }

            if let Some(node) = self.get_node(node_id) {
                let node_type = node.node_type.clone();
                let inputs = node.inputs.clone();

                match node_type {
                    NodeType::Constant(_) => {}
                    NodeType::Learnable(_) => {}
                    NodeType::Variable(_) => {
                        if inputs.len() == 1 {
                            if let Some(input_node) = self.nodes.get_mut(&inputs[0]) {
                                let updated = add_grad(input_node.gradient.clone(), gradient.clone())?;
                                input_node.gradient = Some(updated);
                            }
                        }
                    }
                    NodeType::BinaryOp(ref op) => {
                        if inputs.len() == 2 {
                            let left_val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                            let right_val = self.nodes.get(&inputs[1]).and_then(|n| n.value.clone());

                            match op.as_str() {
                                "add" => {
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                        left_node.gradient = Some(add_grad(left_node.gradient.clone(), gradient.clone())?);
                                    }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                        right_node.gradient = Some(add_grad(right_node.gradient.clone(), gradient.clone())?);
                                    }
                                }
                                "sub" => {
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                        left_node.gradient = Some(add_grad(left_node.gradient.clone(), gradient.clone())?);
                                    }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                        right_node.gradient = Some(add_grad(right_node.gradient.clone(), negate_value(gradient.clone()))?);
                                    }
                                }
                                "mul" => {
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                        if let Some(r) = right_val.clone() {
                                            let upd = mul_grad(gradient.clone(), r)?;
                                            left_node.gradient = Some(add_grad(left_node.gradient.clone(), upd)?);
                                        }
                                    }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                        if let Some(l) = left_val.clone() {
                                            let upd = mul_grad(gradient.clone(), l)?;
                                            right_node.gradient = Some(add_grad(right_node.gradient.clone(), upd)?);
                                        }
                                    }
                                }
                                "div" => {
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                        if let Some(b) = right_val.clone() {
                                            let upd = mul_grad(gradient.clone(), inv_value(&b)?)?;
                                            left_node.gradient = Some(add_grad(left_node.gradient.clone(), upd)?);
                                        }
                                    }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                        if let (Some(a), Some(b)) = (left_val.clone(), right_val.clone()) {
                                            let b_sq = mul_grad(b.clone(), b.clone())?;
                                            let num = mul_grad(gradient.clone(), a)?;
                                            let frac = div_value(num, b_sq)?;
                                            let neg = negate_value(frac);
                                            right_node.gradient = Some(add_grad(right_node.gradient.clone(), neg)?);
                                        }
                                    }
                                }
                                "pow" => {
                                    if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                        if let (Some(a), Some(b)) = (left_val.clone(), right_val.clone()) {
                                            let b_minus_one = add_const(&b, -1.0)?;
                                            let a_pow = pow_value(&a, &b_minus_one)?;
                                            let local = mul_grad(gradient.clone(), mul_grad(b.clone(), a_pow)?)?;
                                            left_node.gradient = Some(add_grad(left_node.gradient.clone(), local)?);
                                        }
                                    }
                                    if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                        if let (Some(a), Some(b)) = (left_val.clone(), right_val.clone()) {
                                            let pow_ab = pow_value(&a, &b)?;
                                            let ln_a = ln_value(&a)?;
                                            let local = mul_grad(gradient.clone(), mul_grad(pow_ab, ln_a)?)?;
                                            right_node.gradient = Some(add_grad(right_node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                "mod" => {}
                                "and" | "or" => {}
                                _ => {}
                            }
                        }
                    }
                    NodeType::UnaryOp(ref op) => {
                        if inputs.len() == 1 {
                            match op.as_str() {
                                "neg" => {
                                    if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                        node.gradient = Some(add_grad(node.gradient.clone(), negate_value(gradient.clone()))?);
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                    NodeType::FunctionCall(name) => {
                        if inputs.len() == 1 {
                            let val = self.nodes.get(&inputs[0]).and_then(|n| n.value.clone());
                            match name.as_str() {
                                "sigmoid" => {
                                    if let Some(Value::Scalar(v)) = val {
                                        let s = 1.0 / (1.0 + (-v).exp());
                                        let g = gradient.as_scalar().ok_or("Expected scalar gradient for scalar sigmoid")?;
                                        let local = Value::Scalar(g * s * (1.0 - s));
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    } else if let Some(Value::Tensor(t)) = val {
                                        let deriv: Vec<f64> = t.data.iter().map(|x| {
                                            let s = 1.0 / (1.0 + (-x).exp());
                                            s * (1.0 - s)
                                        }).collect();
                                        let local = match gradient.clone() {
                                            Value::Tensor(g) => {
                                                if g.shape != t.shape {
                                                    return Err("Sigmoid tensor gradient shape mismatch".to_string());
                                                }
                                                Value::Tensor(Tensor { data: g.data.iter().zip(deriv.iter()).map(|(g, d)| g * d).collect(), shape: g.shape })
                                            }
                                            Value::Scalar(s) => Value::Tensor(Tensor { data: deriv.iter().map(|d| s * d).collect(), shape: t.shape.clone() }),
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                "relu" => {
                                    if let Some(Value::Scalar(v)) = val {
                                        let local = if v > 0.0 { gradient.clone() } else { Value::Scalar(0.0) };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    } else if let Some(Value::Tensor(t)) = val {
                                        let mask: Vec<f64> = t.data.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
                                        let local = match gradient.clone() {
                                            Value::Tensor(g) => {
                                                if g.shape != t.shape {
                                                    return Err("ReLU tensor gradient shape mismatch".to_string());
                                                }
                                                Value::Tensor(Tensor { data: g.data.iter().zip(mask.iter()).map(|(g, m)| g * m).collect(), shape: g.shape })
                                            }
                                            Value::Scalar(s) => Value::Tensor(Tensor { data: mask.iter().map(|m| s * m).collect(), shape: t.shape.clone() }),
                                        };
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(add_grad(node.gradient.clone(), local)?);
                                        }
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn optimize_step(&mut self, learning_rate: f64) -> Result<(), String> {
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();

        for node_id in node_ids {
            if let Some(node) = self.nodes.get(&node_id) {
                if let NodeType::Learnable(_) = &node.node_type {
                    if let (Some(value), Some(gradient)) = (node.value.clone(), node.gradient.clone()) {
                        let updated = match (value, gradient) {
                            (Value::Scalar(v), Value::Scalar(g)) => Value::Scalar(v - learning_rate * g),
                            (Value::Tensor(v), Value::Tensor(g)) => {
                                if v.shape != g.shape {
                                    return Err("Gradient/value tensor shape mismatch".to_string());
                                }
                                let data = v.data.iter().zip(g.data.iter()).map(|(v, g)| v - learning_rate * g).collect();
                                Value::Tensor(Tensor { data, shape: v.shape })
                            }
                            _ => return Err("Mixed scalar/tensor optimization not supported".to_string()),
                        };

                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            let zero = updated.zeros_like();
                            node.value = Some(updated);
                            node.gradient = Some(zero);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    pub fn reset_gradients(&mut self) {
        for node in self.nodes.values_mut() {
            if let Some(val) = node.value.clone() {
                node.gradient = Some(val.zeros_like());
            }
        }
    }
}

impl Default for ComputationalGraph {
    fn default() -> Self {
        Self::new()
    }
}

fn add_grad(current: Option<Value>, delta: Value) -> Result<Value, String> {
    match (current, delta) {
        (None, d) => Ok(d),
        (Some(Value::Scalar(a)), Value::Scalar(b)) => Ok(Value::Scalar(a + b)),
        (Some(Value::Tensor(t1)), Value::Tensor(t2)) => {
            if t1.shape != t2.shape {
                return Err("Gradient tensor shape mismatch".to_string());
            }
            let data = t1.data.iter().zip(t2.data.iter()).map(|(a, b)| a + b).collect();
            Ok(Value::Tensor(Tensor { data, shape: t1.shape }))
        }
        _ => Err("Mixed scalar/tensor gradients not supported".to_string()),
    }
}

fn mul_grad(a: Value, b: Value) -> Result<Value, String> {
    a.map2(&b, |x, y| x * y)
}

fn div_value(a: Value, b: Value) -> Result<Value, String> {
    a.map2(&b, |x, y| x / y)
}

fn negate_value(v: Value) -> Value {
    match v {
        Value::Scalar(s) => Value::Scalar(-s),
        Value::Tensor(t) => Value::Tensor(Tensor { data: t.data.iter().map(|x| -x).collect(), shape: t.shape }),
    }
}

fn inv_value(v: &Value) -> Result<Value, String> {
    match v {
        Value::Scalar(s) => Ok(Value::Scalar(1.0 / s)),
        Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| 1.0 / x).collect(), shape: t.shape.clone() })),
    }
}

fn pow_value(a: &Value, b: &Value) -> Result<Value, String> {
    a.map2(b, |x, y| x.powf(y))
}

fn add_const(v: &Value, c: f64) -> Result<Value, String> {
    match v {
        Value::Scalar(s) => Ok(Value::Scalar(s + c)),
        Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| x + c).collect(), shape: t.shape.clone() })),
    }
}

fn ln_value(v: &Value) -> Result<Value, String> {
    match v {
        Value::Scalar(s) => Ok(Value::Scalar(s.ln())),
        Value::Tensor(t) => Ok(Value::Tensor(Tensor { data: t.data.iter().map(|x| x.ln()).collect(), shape: t.shape.clone() })),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_constant() {
        let mut graph = ComputationalGraph::new();
        let id = graph.add_constant(5.0);
        assert_eq!(graph.get_node(id).map(|n| &n.node_type), Some(&NodeType::Constant(Value::Scalar(5.0))));
    }

    #[test]
    fn test_add_learnable() {
        let mut graph = ComputationalGraph::new();
        let id = graph.add_learnable("x".to_string(), 2.0);
        assert!(matches!(graph.get_node(id).map(|n| &n.node_type), Some(NodeType::Learnable(_))));
    }

    #[test]
    fn test_binary_operation() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(3.0);
        let b = graph.add_constant(2.0);
        let result = graph.add_binary_op("add", a, b);
        assert_eq!(graph.get_node(result).map(|n| n.inputs.len()), Some(2));
    }

    #[test]
    fn test_forward_pass() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(3.0);
        let b = graph.add_constant(2.0);
        let _sum = graph.add_binary_op("add", a, b);

        graph.forward_pass().unwrap();

        assert_eq!(graph.get_node(a).and_then(|n| n.value.clone()), Some(Value::Scalar(3.0)));
        assert_eq!(graph.get_node(b).and_then(|n| n.value.clone()), Some(Value::Scalar(2.0)));
    }

    #[test]
    fn test_backward_pass_simple() {
        let mut graph = ComputationalGraph::new();

        let x = graph.add_learnable("x".to_string(), 3.0);
        let y = graph.add_binary_op("mul", x, x);

        graph.forward_pass().unwrap();
        assert_eq!(graph.get_node(y).and_then(|n| n.value.clone()), Some(Value::Scalar(9.0)));

        graph.backward_pass(y).unwrap();

        let x_gradient = graph.get_node(x).and_then(|n| n.gradient.clone());
        assert!(x_gradient.is_some());
        match x_gradient.unwrap() {
            Value::Scalar(g) => assert!((g - 6.0).abs() < 0.0001),
            _ => panic!("expected scalar gradient"),
        }
    }

    #[test]
    fn test_gradient_descent() {
        let mut graph = ComputationalGraph::new();

        let x = graph.add_learnable("x".to_string(), 5.0);
        let y = graph.add_binary_op("mul", x, x);

        for _ in 0..10 {
            graph.forward_pass().unwrap();
            graph.backward_pass(y).unwrap();
            graph.optimize_step(0.1).unwrap();
            graph.reset_gradients();
        }

        let final_x = graph.get_node(x).and_then(|n| n.value.clone());
        assert!(final_x.is_some());
        if let Value::Scalar(v) = final_x.unwrap() {
            assert!(v.abs() < 5.0);
        } else {
            panic!("expected scalar value");
        }
    }

    #[test]
    fn test_mod_and_or_forward() {
        let mut graph = ComputationalGraph::new();
        let a = graph.add_constant(5.0);
        let b = graph.add_constant(2.0);
        let m = graph.add_binary_op("mod", a, b);
        let o = graph.add_binary_op("or", a, b);
        let z = graph.add_constant(0.0);
        let a2 = graph.add_binary_op("and", a, z);

        graph.forward_pass().unwrap();
        assert_eq!(graph.get_node(m).and_then(|n| n.value.clone()), Some(Value::Scalar(1.0)));
        assert_eq!(graph.get_node(o).and_then(|n| n.value.clone()), Some(Value::Scalar(1.0)));
        assert_eq!(graph.get_node(a2).and_then(|n| n.value.clone()), Some(Value::Scalar(0.0)));
    }

    #[test]
    fn test_tensor_forward_backward() {
        let mut graph = ComputationalGraph::new();
        let t = graph.add_constant_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let l = graph.add_learnable_tensor("w".to_string(), vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).unwrap();
        let m = graph.add_binary_op("mul", t, l);

        graph.forward_pass().unwrap();
        if let Some(Value::Tensor(val)) = graph.get_node(m).and_then(|n| n.value.clone()) {
            assert_eq!(val.data, vec![1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("expected tensor value");
        }

        graph.backward_pass(m).unwrap();
        if let Some(Value::Tensor(g)) = graph.get_node(l).and_then(|n| n.gradient.clone()) {
            assert_eq!(g.data, vec![1.0, 2.0, 3.0, 4.0]);
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn test_tensor_broadcast_add() {
        let mut graph = ComputationalGraph::new();
        let t = graph.add_constant_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let c = graph.add_constant(1.5);
        let s = graph.add_binary_op("add", t, c);

        graph.forward_pass().unwrap();
        if let Some(Value::Tensor(val)) = graph.get_node(s).and_then(|n| n.value.clone()) {
            assert_eq!(val.data, vec![2.5, 3.5, 4.5, 5.5]);
            assert_eq!(val.shape, vec![2, 2]);
        } else {
            panic!("expected tensor value");
        }
    }

    #[test]
    fn test_tensor_sigmoid_backward() {
        let mut graph = ComputationalGraph::new();
        let x = graph.add_learnable_tensor("x".to_string(), vec![0.0, 1.0, -1.0, 2.0], vec![2, 2]).unwrap();
        let y = graph.add_function_call("sigmoid".to_string(), vec![x]);

        graph.forward_pass().unwrap();
        graph.backward_pass(y).unwrap();

        // gradient at x should be sigmoid(x) * (1 - sigmoid(x))
        if let Some(Value::Tensor(g)) = graph.get_node(x).and_then(|n| n.gradient.clone()) {
            let xs: Vec<f64> = vec![0.0_f64, 1.0_f64, -1.0_f64, 2.0_f64];
            let s: Vec<f64> = xs.iter().map(|&v| 1.0_f64 / (1.0_f64 + (-v).exp())).collect();
            let expected: Vec<f64> = s.iter().map(|&sv| sv * (1.0 - sv)).collect();
            for (a, b) in g.data.iter().zip(expected.iter()) {
                assert!((a - b).abs() < 1e-9);
            }
        } else {
            panic!("expected tensor gradient");
        }
    }

    #[test]
    fn test_backward_linear_affine() {
        // y = (x * 2) + 5 => dy/dx = 2
        let mut graph = ComputationalGraph::new();
        let x = graph.add_learnable("x".to_string(), 3.0);
        let c2 = graph.add_constant(2.0);
        let c5 = graph.add_constant(5.0);
        let mul = graph.add_binary_op("mul", x, c2);
        let y = graph.add_binary_op("add", mul, c5);

        graph.forward_pass().unwrap();
        graph.backward_pass(y).unwrap();

        let x_gradient = graph.get_node(x).and_then(|n| n.gradient.clone());
        assert!(x_gradient.is_some());
        match x_gradient.unwrap() {
            Value::Scalar(g) => assert!((g - 2.0).abs() < 1e-9),
            _ => panic!("expected scalar gradient"),
        }
    }
}
