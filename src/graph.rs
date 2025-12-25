use std::collections::HashMap;
use crate::ast::{BinaryOperator, Expression, UnaryOperator};

/// A unique identifier for a node in the computational graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new(id: usize) -> Self {
        NodeId(id)
    }
}

/// Represents a node in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub id: NodeId,
    pub node_type: NodeType,
    pub inputs: Vec<NodeId>, // Input dependencies
    pub value: Option<f64>,  // Current value
    pub gradient: Option<f64>, // Gradient for backprop
}

/// Represents an operation/node in the computational graph
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    /// Constant value
    Constant(f64),
    /// Learnable variable (declared with 'learn')
    Learnable(String),
    /// Regular variable (declared with 'let')
    Variable(String),
    /// Binary operation: add, sub, mul, div, pow, eq, ne, lt, gt, le, ge
    BinaryOp(String),
    /// Unary operation: neg, not
    UnaryOp(String),
    /// Function call: sigmoid, relu, dot, mse, etc.
    FunctionCall(String),
}

/// The computational graph - directed acyclic graph (DAG)
#[derive(Debug, Clone)]
pub struct ComputationalGraph {
    nodes: HashMap<NodeId, Node>,
    next_id: usize,
    learnables: Vec<String>, // Track learnable variables for gradients
}

impl ComputationalGraph {
    pub fn new() -> Self {
        ComputationalGraph {
            nodes: HashMap::new(),
            next_id: 0,
            learnables: Vec::new(),
        }
    }

    /// Add a constant node
    pub fn add_constant(&mut self, value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;

        let node = Node {
            id,
            node_type: NodeType::Constant(value),
            inputs: Vec::new(),
            value: Some(value),
            gradient: None,
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a learnable variable node
    pub fn add_learnable(&mut self, name: String, initial_value: f64) -> NodeId {
        let id = NodeId::new(self.next_id);
        self.next_id += 1;
        self.learnables.push(name.clone());

        let node = Node {
            id,
            node_type: NodeType::Learnable(name),
            inputs: Vec::new(),
            value: Some(initial_value),
            gradient: Some(0.0), // Gradients start at zero
        };

        self.nodes.insert(id, node);
        id
    }

    /// Add a regular variable node
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

    /// Add a binary operation node
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

    /// Add a unary operation node
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

    /// Add a function call node
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

    /// Convert an AST expression into a computational graph
    pub fn build_from_expression(&mut self, expr: &Expression, variables: &HashMap<String, NodeId>) -> Result<NodeId, String> {
        match expr {
            Expression::Number(n) => Ok(self.add_constant(*n)),
            Expression::Identifier(name) => {
                variables.get(name)
                    .copied()
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }
            Expression::BinaryOp { left, op, right } => {
                let left_id = self.build_from_expression(left, variables)?;
                let right_id = self.build_from_expression(right, variables)?;
                
                let op_str = match op {
                    BinaryOperator::Add => "add",
                    BinaryOperator::Sub => "sub",
                    BinaryOperator::Mul => "mul",
                    BinaryOperator::Div => "div",
                    BinaryOperator::Pow => "pow",
                    BinaryOperator::Equal => "eq",
                    BinaryOperator::NotEqual => "ne",
                    BinaryOperator::Less => "lt",
                    BinaryOperator::Greater => "gt",
                    BinaryOperator::LessEq => "le",
                    BinaryOperator::GreaterEq => "ge",
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

    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> Option<&mut Node> {
        self.nodes.get_mut(&id)
    }

    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, Node> {
        &self.nodes
    }

    /// Get learnable variables
    pub fn learnables(&self) -> &[String] {
        &self.learnables
    }

    /// Evaluate the forward pass
    pub fn forward_pass(&mut self) -> Result<(), String> {
        // Collect node data to iterate over
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();
        
        for node_id in node_ids {
            let node_type = self.nodes.get(&node_id).map(|n| &n.node_type).cloned();
            let inputs = self.nodes.get(&node_id).map(|n| n.inputs.clone()).unwrap_or_default();
            
            if let Some(node_type) = node_type {
                match node_type {
                    NodeType::Constant(v) => {
                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.value = Some(v);
                        }
                    }
                    NodeType::Learnable(_, ) => {
                        // Already initialized
                    }
                    NodeType::Variable(_, ) => {
                        if inputs.len() == 1 {
                            if let Some(input_val) = self.nodes.get(&inputs[0]).and_then(|n| n.value) {
                                if let Some(node) = self.nodes.get_mut(&node_id) {
                                    node.value = Some(input_val);
                                }
                            }
                        }
                    }
                    NodeType::BinaryOp(op) => {
                        if inputs.len() == 2 {
                            let left_val = self.nodes.get(&inputs[0])
                                .and_then(|n| n.value)
                                .ok_or("Missing left operand")?;
                            let right_val = self.nodes.get(&inputs[1])
                                .and_then(|n| n.value)
                                .ok_or("Missing right operand")?;

                            let result = match op.as_str() {
                                "add" => left_val + right_val,
                                "sub" => left_val - right_val,
                                "mul" => left_val * right_val,
                                "div" => left_val / right_val,
                                "pow" => left_val.powf(right_val),
                                _ => return Err(format!("Unknown binary op: {}", op)),
                            };
                            
                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::UnaryOp(op) => {
                        if inputs.len() == 1 {
                            let val = self.nodes.get(&inputs[0])
                                .and_then(|n| n.value)
                                .ok_or("Missing operand")?;

                            let result = match op.as_str() {
                                "neg" => -val,
                                "not" => if val != 0.0 { 0.0 } else { 1.0 },
                                _ => return Err(format!("Unknown unary op: {}", op)),
                            };
                            
                            if let Some(node) = self.nodes.get_mut(&node_id) {
                                node.value = Some(result);
                            }
                        }
                    }
                    NodeType::FunctionCall(name) => {
                        // Placeholder for function calls
                        match name.as_str() {
                            "sigmoid" => {
                                if inputs.len() == 1 {
                                    let val = self.nodes.get(&inputs[0])
                                        .and_then(|n| n.value)
                                        .ok_or("Missing argument")?;
                                    let result = 1.0 / (1.0 + (-val).exp());
                                    if let Some(node) = self.nodes.get_mut(&node_id) {
                                        node.value = Some(result);
                                    }
                                }
                            }
                            "relu" => {
                                if inputs.len() == 1 {
                                    let val = self.nodes.get(&inputs[0])
                                        .and_then(|n| n.value)
                                        .ok_or("Missing argument")?;
                                    let result = if val > 0.0 { val } else { 0.0 };
                                    if let Some(node) = self.nodes.get_mut(&node_id) {
                                        node.value = Some(result);
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Print the graph structure for debugging
    pub fn print_structure(&self) {
        println!("=== Computational Graph ===");
        for (id, node) in &self.nodes {
            println!(
                "Node {:?}: {:?}, value: {:?}, gradients: {:?}, inputs: {:?}",
                id, node.node_type, node.value, node.gradient, node.inputs
            );
        }
    }

    /// Backward pass - compute gradients via reverse-mode autodiff
    pub fn backward_pass(&mut self, output_id: NodeId) -> Result<(), String> {
        // Initialize gradient of output node to 1.0 (dL/dL = 1)
        if let Some(node) = self.nodes.get_mut(&output_id) {
            node.gradient = Some(1.0);
        }

        // Collect node IDs in reverse topological order (for backprop)
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();

        // Backpropagate through each node
        for node_id in node_ids {
            if let Some(gradient) = self.get_node(node_id).and_then(|n| n.gradient) {
                if gradient == 0.0 {
                    continue;
                }

                if let Some(node) = self.get_node(node_id) {
                    let node_type = node.node_type.clone();
                    let inputs = node.inputs.clone();

                    match node_type {
                        NodeType::Constant(_) => {
                            // No gradient for constants
                        }
                        NodeType::Learnable(_, ) => {
                            // Gradient already set, will be used by optimizer
                        }
                        NodeType::Variable(_, ) => {
                            // Pass gradient to input
                            if inputs.len() == 1 {
                                if let Some(input_node) = self.nodes.get_mut(&inputs[0]) {
                                    input_node.gradient = Some(input_node.gradient.unwrap_or(0.0) + gradient);
                                }
                            }
                        }
                        NodeType::BinaryOp(ref op) => {
                            if inputs.len() == 2 {
                                let left_val = self.nodes.get(&inputs[0]).and_then(|n| n.value);
                                let right_val = self.nodes.get(&inputs[1]).and_then(|n| n.value);

                                match op.as_str() {
                                    "add" => {
                                        // d(a+b)/da = 1, d(a+b)/db = 1
                                        if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                            left_node.gradient = Some(left_node.gradient.unwrap_or(0.0) + gradient);
                                        }
                                        if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                            right_node.gradient = Some(right_node.gradient.unwrap_or(0.0) + gradient);
                                        }
                                    }
                                    "sub" => {
                                        // d(a-b)/da = 1, d(a-b)/db = -1
                                        if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                            left_node.gradient = Some(left_node.gradient.unwrap_or(0.0) + gradient);
                                        }
                                        if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                            right_node.gradient = Some(right_node.gradient.unwrap_or(0.0) - gradient);
                                        }
                                    }
                                    "mul" => {
                                        // d(a*b)/da = b, d(a*b)/db = a
                                        if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                            if let Some(b) = right_val {
                                                left_node.gradient = Some(left_node.gradient.unwrap_or(0.0) + gradient * b);
                                            }
                                        }
                                        if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                            if let Some(a) = left_val {
                                                right_node.gradient = Some(right_node.gradient.unwrap_or(0.0) + gradient * a);
                                            }
                                        }
                                    }
                                    "div" => {
                                        // d(a/b)/da = 1/b, d(a/b)/db = -a/b²
                                        if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                            if let Some(b) = right_val {
                                                if b != 0.0 {
                                                    left_node.gradient = Some(left_node.gradient.unwrap_or(0.0) + gradient / b);
                                                }
                                            }
                                        }
                                        if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                            if let Some(a) = left_val {
                                                if let Some(b) = right_val {
                                                    if b != 0.0 {
                                                        right_node.gradient = Some(right_node.gradient.unwrap_or(0.0) - gradient * a / (b * b));
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    "pow" => {
                                        // d(a^b)/da = b*a^(b-1), d(a^b)/db = a^b*ln(a)
                                        if let Some(left_node) = self.nodes.get_mut(&inputs[0]) {
                                            if let Some(a) = left_val {
                                                if let Some(b) = right_val {
                                                    if a > 0.0 {
                                                        let local_grad = gradient * b * a.powf(b - 1.0);
                                                        left_node.gradient = Some(left_node.gradient.unwrap_or(0.0) + local_grad);
                                                    }
                                                }
                                            }
                                        }
                                        if let Some(right_node) = self.nodes.get_mut(&inputs[1]) {
                                            if let Some(a) = left_val {
                                                if a > 0.0 {
                                                    let local_grad = gradient * a.powf(right_val.unwrap_or(0.0)) * a.ln();
                                                    right_node.gradient = Some(right_node.gradient.unwrap_or(0.0) + local_grad);
                                                }
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        NodeType::UnaryOp(ref op) => {
                            if inputs.len() == 1 {
                                let val = self.nodes.get(&inputs[0]).and_then(|n| n.value);

                                match op.as_str() {
                                    "neg" => {
                                        // d(-a)/da = -1
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            node.gradient = Some(node.gradient.unwrap_or(0.0) - gradient);
                                        }
                                    }
                                    "sigmoid" => {
                                        // d(sigmoid(x))/dx = sigmoid(x)(1-sigmoid(x))
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            if let Some(v) = val {
                                                let s = 1.0 / (1.0 + (-v).exp());
                                                let local_grad = gradient * s * (1.0 - s);
                                                node.gradient = Some(node.gradient.unwrap_or(0.0) + local_grad);
                                            }
                                        }
                                    }
                                    "relu" => {
                                        // d(relu(x))/dx = 1 if x > 0 else 0
                                        if let Some(node) = self.nodes.get_mut(&inputs[0]) {
                                            if let Some(v) = val {
                                                let local_grad = if v > 0.0 { gradient } else { 0.0 };
                                                node.gradient = Some(node.gradient.unwrap_or(0.0) + local_grad);
                                            }
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        NodeType::FunctionCall(_) => {
                            // TODO: Implement gradients for other functions
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Optimize learnable variables using SGD
    pub fn optimize_step(&mut self, learning_rate: f64) -> Result<(), String> {
        let node_ids: Vec<NodeId> = self.nodes.keys().copied().collect();

        for node_id in node_ids {
            if let Some(node) = self.nodes.get(&node_id) {
                if let NodeType::Learnable(_) = &node.node_type {
                    if let (Some(value), Some(gradient)) = (node.value, node.gradient) {
                        // SGD: x = x - learning_rate * gradient
                        if let Some(node) = self.nodes.get_mut(&node_id) {
                            node.value = Some(value - learning_rate * gradient);
                            // Reset gradient for next iteration
                            node.gradient = Some(0.0);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Reset all gradients to zero
    pub fn reset_gradients(&mut self) {
        for node in self.nodes.values_mut() {
            node.gradient = Some(0.0);
        }
    }
}

impl Default for ComputationalGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_constant() {
        let mut graph = ComputationalGraph::new();
        let id = graph.add_constant(5.0);
        assert_eq!(graph.get_node(id).map(|n| &n.node_type), Some(&NodeType::Constant(5.0)));
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

        assert_eq!(graph.get_node(a).and_then(|n| n.value), Some(3.0));
        assert_eq!(graph.get_node(b).and_then(|n| n.value), Some(2.0));
    }

    #[test]
    fn test_backward_pass_simple() {
        let mut graph = ComputationalGraph::new();
        
        // Create: y = x^2
        let x = graph.add_learnable("x".to_string(), 3.0);
        let y = graph.add_binary_op("mul", x, x); // x * x = x^2
        
        // Forward pass
        graph.forward_pass().unwrap();
        assert_eq!(graph.get_node(y).and_then(|n| n.value), Some(9.0)); // 3^2 = 9
        
        // Backward pass
        graph.backward_pass(y).unwrap();
        
        // dy/dx should be 2*x = 6
        let x_gradient = graph.get_node(x).and_then(|n| n.gradient);
        assert!(x_gradient.is_some());
        assert!((x_gradient.unwrap() - 6.0).abs() < 0.0001);
    }

    #[test]
    fn test_gradient_descent() {
        let mut graph = ComputationalGraph::new();
        
        // Start with x = 5.0, want to minimize y = x^2
        let x = graph.add_learnable("x".to_string(), 5.0);
        let y = graph.add_binary_op("mul", x, x);
        
        // Do a few gradient descent steps
        for _ in 0..10 {
            graph.forward_pass().unwrap();
            graph.backward_pass(y).unwrap();
            graph.optimize_step(0.1).unwrap(); // learning rate = 0.1
            graph.reset_gradients();
        }
        
        // x should be closer to 0
        let final_x = graph.get_node(x).and_then(|n| n.value);
        assert!(final_x.is_some());
        assert!(final_x.unwrap().abs() < 5.0); // Should have reduced from 5.0
    }
}
