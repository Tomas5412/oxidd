//! Recursive single-threaded apply algorithms

use std::borrow::Borrow;

use fixedbitset::FixedBitSet;

use oxidd_core::function::{EdgeOfFunc, Function, INodeOfFunc, NumberBase, PseudoBooleanFunction};
use oxidd_core::util::{AllocResult, Borrowed, EdgeDropGuard};
use oxidd_core::{
    ApplyCache, Edge, HasApplyCache, HasLevel, InnerNode, LevelNo, Manager, Node, Tag, VarNo,
};
use oxidd_derive::Function;
use oxidd_dump::dot::DotStyle;

#[cfg(feature = "statistics")]
use super::STAT_COUNTERS;
use super::{MTBDDOp, Operation, collect_children, reduce, stat};

// spell-checker:ignore fnode,gnode,hnode,vnode,flevel,glevel,hlevel,vlevel

/// Recursively apply the binary operator `OP` to `f` and `g`
///
/// We use a `const` parameter `OP` to have specialized version of this function
/// for each operator.
fn apply_bin<M, T, const OP: u8>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call OP);
    let (operator, op1, op2) = match super::terminal_bin::<M, T, OP>(manager, &f, &g)? {
        Operation::Binary(o, op1, op2) => (o, op1, op2),
        Operation::Done(h) => return Ok(h),
    };

    // Query apply cache
    stat!(cache_query OP);
    if let Some(h) = manager
        .apply_cache()
        .get(manager, operator, &[op1.borrowed(), op2.borrowed()])
    {
        stat!(cache_hit OP);
        return Ok(h);
    }

    let fnode = manager.get_node(&f);
    let gnode = manager.get_node(&g);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let level = std::cmp::min(flevel, glevel);

    // Collect cofactors of all top-most nodes
    let (f0, f1) = if flevel == level {
        collect_children(fnode.unwrap_inner())
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (g0, g1) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g.borrowed(), g.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_bin::<M, T, OP>(manager, f0, g0)?);
    let e = EdgeDropGuard::new(manager, apply_bin::<M, T, OP>(manager, f1, g1)?);
    let h = reduce(manager, level, t.into_edge(), e.into_edge(), operator)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, operator, &[op1, op2], h.borrowed());

    Ok(h)
}

/// Recursively restrict a set of `vars` (a conjunction of literals) to
/// constant values in `f`
fn restrict<M, T>(
    manager: &M,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Restrict);

    let (Node::Inner(fnode), Node::Inner(vnode)) = (manager.get_node(&f), manager.get_node(&vars))
    else {
        return Ok(manager.clone_edge(&f));
    };

    enum InnerResult<'a, M: Manager> {
        Done(M::Edge),
        Rec {
            vars: Borrowed<'a, M::Edge>,
            f: Borrowed<'a, M::Edge>,
            fnode: &'a M::InnerNode,
        },
    }

    /// Tail-recursive part of [`restrict()`]. `f` is the function of which the
    /// variables should be restricted to constant values according to `vars`.
    ///
    /// Invariant: `f` points to `fnode` at `flevel`, `vars` points to `vnode`
    #[inline]
    fn inner<'a, M, T>(
        manager: &'a M,
        f: Borrowed<'a, M::Edge>,
        fnode: &'a M::InnerNode,
        flevel: LevelNo,
        vars: Borrowed<'a, M::Edge>,
        vnode: &'a M::InnerNode,
    ) -> InnerResult<'a, M>
    where
        M: Manager<Terminal = T>,
        M::InnerNode: HasLevel,
        T: NumberBase,
    {
        debug_assert!(std::ptr::eq(manager.get_node(&f).unwrap_inner(), fnode));
        debug_assert_eq!(fnode.level(), flevel);
        debug_assert!(std::ptr::eq(manager.get_node(&vars).unwrap_inner(), vnode));

        let vlevel = vnode.level();
        if vlevel > flevel {
            // f above vars
            return InnerResult::Rec { vars, f, fnode };
        }

        let vt = vnode.child(0);
        if vlevel < flevel {
            // vars above f
            return match manager.get_node(&vt) {
                Node::Inner(n) => inner(manager, f, fnode, flevel, vt, n),
                Node::Terminal(t) if t.borrow().is_one() => {
                    InnerResult::Done(manager.clone_edge(&f))
                }
                Node::Terminal(_) => {
                    let ve = vnode.child(1);
                    if let Node::Inner(n) = manager.get_node(&ve) {
                        inner(manager, f, fnode, flevel, ve, n)
                    } else {
                        InnerResult::Done(manager.clone_edge(&f))
                    }
                }
            };
        }

        debug_assert_eq!(vlevel, flevel);
        // top var at the level of f ⇒ select accordingly
        let (f, vars, vnode) = match manager.get_node(&vt) {
            Node::Inner(n) => {
                debug_assert!(
                    matches!(manager.get_node(&vnode.child(1)), Node::Terminal(t) if t.borrow().is_zero()),
                    "vars must be a conjunction of literals"
                );
                // positive literal ⇒ select then branch
                (fnode.child(0), vt, n)
            }
            Node::Terminal(t) if t.borrow().is_one() => {
                debug_assert!(
                    matches!(manager.get_node(&vnode.child(1)), Node::Terminal(t) if t.borrow().is_zero()),
                    "vars must be a conjunction of literals"
                );
                // positive literal ⇒ select then branch
                return InnerResult::Done(manager.clone_edge(&fnode.child(0)));
            }
            Node::Terminal(_) => {
                // negative literal ⇒ select else branch
                let f = fnode.child(1);
                let ve = vnode.child(1);
                if let Node::Inner(n) = manager.get_node(&ve) {
                    (f, ve, n)
                } else {
                    return InnerResult::Done(manager.clone_edge(&f));
                }
            }
        };

        if let Node::Inner(fnode) = manager.get_node(&f) {
            inner(manager, f, fnode, fnode.level(), vars, vnode)
        } else {
            InnerResult::Done(manager.clone_edge(&f))
        }
    }

    match inner(manager, f, fnode, fnode.level(), vars, vnode) {
        InnerResult::Done(res) => Ok(res),
        InnerResult::Rec { vars, f, fnode } => {
            // f above top-most restrict variable

            // Query apply cache
            stat!(cache_query MTBDDOp::Restrict);
            if let Some(res) = manager.apply_cache().get(
                manager,
                MTBDDOp::Restrict,
                &[f.borrowed(), vars.borrowed()],
            ) {
                stat!(cache_hit MTBDDOp::Restrict);
                return Ok(res);
            }

            let (ft, fe) = collect_children(fnode);
            let t = EdgeDropGuard::new(manager, restrict(manager, ft, vars.borrowed())?);
            let e = EdgeDropGuard::new(manager, restrict(manager, fe, vars.borrowed())?);
            let res = reduce(
                manager,
                fnode.level(),
                t.into_edge(),
                e.into_edge(),
                MTBDDOp::Restrict,
            )?;

            manager
                .apply_cache()
                .add(manager, MTBDDOp::Restrict, &[f, vars], res.borrowed());

            Ok(res)
        }
    }
}

/// Project a set of variables 'vars' (a conjunction of literals)
/// on f by merging all branches of vars
fn project<M, T>(
    manager: &M,
    f: Borrowed<M::Edge>,
    vars: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Restrict);
    

    let Node::Inner(vnode) = manager.get_node(&vars) else {
        return Ok(manager.clone_edge(&f));
    };

    /// Get the next iteration of a conjunction of literals.
    /// This is done because project's 'vars' parameter is the same one as restrict. So to get the next iteration a couple of things must be checked.
    /// This is defined inside of the function because I didn't want to bloat the file with functions other than the basic ones.
    /// This multiple definition could be a huge waste of memory. 
    // Best case, 'project' should be modified such that we don't need to care for definitions the function does not use.
    #[inline]
    fn pick_next<'a, M, T>(
        manager: &'a M, 
        vnode: &'a M::InnerNode
    ) -> Borrowed<'a, M::Edge>
    where
        M: Manager<Terminal = T>,
        T: NumberBase,
    {
        let vt = vnode.child(0);
        match manager.get_node(&vt) {
            Node::Inner(_) => vt,
            Node::Terminal(t) if t.borrow().is_one() => vt,
            _ => vnode.child(1),
        }
    }


    let fnode = match manager.get_node(&f) {
        Node::Inner(n) => n,
        Node::Terminal(_) => {
            // f is terminal; double the result
            let next = pick_next(manager, vnode);
            let p = EdgeDropGuard::new(manager, project::<_, T>(manager, f.borrowed(), next)?);
            return apply_bin::<_, T, {MTBDDOp::Add as u8}>(manager, p.borrowed(), p.borrowed());
        }
    };

    let flevel = fnode.level();
    let vlevel = vnode.level();

    if vlevel < flevel {
        // f above vars; select the next iteration 
        let next = pick_next(manager, vnode);
        let p = EdgeDropGuard::new(manager, project::<_, T>(manager, f.borrowed(), next)?);
        return apply_bin::<_, T, {MTBDDOp::Add as u8}>(manager, p.borrowed(), p.borrowed());
    }

    // Query apply cache.
    stat!(cache_query MTBDDOp::Project);
    if let Some(res) =
        manager
            .apply_cache()
            .get(manager, MTBDDOp::Project, &[f.borrowed(), vars.borrowed()])
    {
        stat!(cache_hit MTBDDOp::Project);
        return Ok(res);
    }

    let res = if vlevel > flevel {
        // vars above f; continue normally
        let (ft, fe) = collect_children(fnode);
        let t = EdgeDropGuard::new(manager, project::<_, T>(manager, ft, vars.borrowed())?);
        let e = EdgeDropGuard::new(manager, project::<_, T>(manager, fe, vars.borrowed())?);
        reduce(manager, 
            flevel, 
            t.into_edge(), 
            e.into_edge(), 
            MTBDDOp::Project)?
    } else {
        // top var at the level of f; add both branches, and continue.
        let (ft, fe) = collect_children(fnode);
        let next = pick_next(manager, vnode);
        let p1 = EdgeDropGuard::new(manager, project::<_, T>(manager, ft, next.borrowed())?);
        let p0 = EdgeDropGuard::new(manager, project::<_, T>(manager, fe, next.borrowed())?);
        apply_bin::<_, T, {MTBDDOp::Add as u8}>(manager, p1.borrowed(), p0.borrowed())?
    };

    manager
        .apply_cache()
        .add(manager, MTBDDOp::Project, &[f.borrowed(), vars.borrowed()], res.borrowed());

    Ok(res)
}


/// Recursively apply the if-then-else operator (`if f { g } else { h }`)
///
/// `f` must be a 0-1-valued MTBDD (see [`PseudoBooleanFunction::ite_edge`]).
/// As an extension of the classical restriction, terminals of `f` other than
/// `0` and `1` are treated as "truthy" (`debug_assert`-ed against,
/// since this indicates a violation of the documented precondition).
fn apply_ite<M, T>(
    manager: &M,
    f: Borrowed<M::Edge>,
    g: Borrowed<M::Edge>,
    h: Borrowed<M::Edge>,
) -> AllocResult<M::Edge>
where
    M: Manager<Terminal = T> + HasApplyCache<M, MTBDDOp>,
    M::InnerNode: HasLevel,
    T: NumberBase,
{
    stat!(call MTBDDOp::Ite);

    // The condition is irrelevant if both branches agree.
    if g == h {
        return Ok(manager.clone_edge(&g));
    }

    // Terminal cases for `f`. We decide as soon as `f` resolves to a
    // terminal, which is what makes this a 0-1-valued-condition restricted
    // "ite", as opposed to a fully generic ternary operator.
    let fnode = match manager.get_node(&f) {
        Node::Inner(node) => node,
        Node::Terminal(t) => {
            let t = t.borrow();
            return Ok(if t.is_zero() {
                manager.clone_edge(&h)
            } else {
                debug_assert!(t.is_one(), "the condition of `ite` must be 0-1-valued");
                manager.clone_edge(&g)
            });
        }
    };

    // Query apply cache
    stat!(cache_query MTBDDOp::Ite);
    if let Some(res) = manager.apply_cache().get(
        manager,
        MTBDDOp::Ite,
        &[f.borrowed(), g.borrowed(), h.borrowed()],
    ) {
        stat!(cache_hit MTBDDOp::Ite);
        return Ok(res);
    }

    let gnode = manager.get_node(&g);
    let hnode = manager.get_node(&h);
    let flevel = fnode.level();
    let glevel = gnode.level();
    let hlevel = hnode.level();
    let level = flevel.min(glevel).min(hlevel);

    // Collect cofactors of all top-most nodes
    let (ft, fe) = if flevel == level {
        collect_children(fnode)
    } else {
        (f.borrowed(), f.borrowed())
    };
    let (gt, ge) = if glevel == level {
        collect_children(gnode.unwrap_inner())
    } else {
        (g.borrowed(), g.borrowed())
    };
    let (ht, he) = if hlevel == level {
        collect_children(hnode.unwrap_inner())
    } else {
        (h.borrowed(), h.borrowed())
    };

    let t = EdgeDropGuard::new(manager, apply_ite(manager, ft, gt, ht)?);
    let e = EdgeDropGuard::new(manager, apply_ite(manager, fe, ge, he)?);
    let res = reduce(manager, level, t.into_edge(), e.into_edge(), MTBDDOp::Ite)?;

    // Add to apply cache
    manager
        .apply_cache()
        .add(manager, MTBDDOp::Ite, &[f, g, h], res.borrowed());

    Ok(res)
}

// --- Function Interface ------------------------------------------------------

/// Workaround for https://github.com/rust-lang/rust/issues/49601
trait HasMTBDDOpApplyCache<M: Manager>: HasApplyCache<M, MTBDDOp> {}
impl<M: Manager + HasApplyCache<M, MTBDDOp>> HasMTBDDOpApplyCache<M> for M {}

/// Boolean function backed by a binary decision diagram
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Function, Debug)]
#[repr_id = "MTBDD"]
#[repr(transparent)]
pub struct MTBDDFunction<F: Function>(F);

impl<F: Function> From<F> for MTBDDFunction<F> {
    #[inline(always)]
    fn from(value: F) -> Self {
        MTBDDFunction(value)
    }
}

impl<F: Function> MTBDDFunction<F> {
    /// Convert `self` into the underlying [`Function`]
    #[inline(always)]
    pub fn into_inner(self) -> F {
        self.0
    }
}

impl<F: Function, T: NumberBase> PseudoBooleanFunction for MTBDDFunction<F>
where
    for<'id> F::Manager<'id>: Manager<Terminal = T> + HasMTBDDOpApplyCache<F::Manager<'id>>,
    for<'id> INodeOfFunc<'id, F>: HasLevel,
{
    type Number = T;

    #[inline]
    fn constant_edge<'id>(
        manager: &Self::Manager<'id>,
        value: Self::Number,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        manager.get_terminal(value)
    }

    #[inline]
    fn var_edge<'id>(
        manager: &Self::Manager<'id>,
        var: VarNo,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        let level = manager.var_to_level(var);
        let t = EdgeDropGuard::new(manager, manager.get_terminal(T::one())?);
        let e = EdgeDropGuard::new(manager, manager.get_terminal(T::zero())?);
        oxidd_core::LevelView::get_or_insert(
            &mut manager.level(level),
            InnerNode::new(level, [t.into_edge(), e.into_edge()]),
        )
    }

    #[inline]
    fn add_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Add as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn sub_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Sub as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn mul_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Mul as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn div_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Div as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn min_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Min as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn max_edge<'id>(
        manager: &Self::Manager<'id>,
        lhs: &EdgeOfFunc<'id, Self>,
        rhs: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_bin::<_, T, { MTBDDOp::Max as u8 }>(manager, lhs.borrowed(), rhs.borrowed())
    }

    #[inline]
    fn restrict_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        restrict::<_, T>(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn project_edge<'id>(
        manager: &Self::Manager<'id>,
        root: &EdgeOfFunc<'id, Self>,
        vars: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        project::<_, T>(manager, root.borrowed(), vars.borrowed())
    }

    #[inline]
    fn ite_edge<'id>(
        manager: &Self::Manager<'id>,
        if_edge: &EdgeOfFunc<'id, Self>,
        then_edge: &EdgeOfFunc<'id, Self>,
        else_edge: &EdgeOfFunc<'id, Self>,
    ) -> AllocResult<EdgeOfFunc<'id, Self>> {
        apply_ite::<_, T>(
            manager,
            if_edge.borrowed(),
            then_edge.borrowed(),
            else_edge.borrowed(),
        )
    }

    #[inline]
    fn eval_edge<'id>(
        manager: &Self::Manager<'id>,
        edge: &EdgeOfFunc<'id, Self>,
        args: impl IntoIterator<Item = (VarNo, bool)>,
    ) -> T {
        // `choices` maps levels to the child number to choose
        let mut choices = FixedBitSet::with_capacity(manager.num_levels() as usize);
        for (var, val) in args {
            // child 0 is "then"/"true", hence the negation
            choices.set(manager.var_to_level(var) as usize, !val);
        }

        #[inline] // this function is tail-recursive
        fn inner<M, T: Clone>(manager: &M, edge: Borrowed<M::Edge>, choices: &FixedBitSet) -> T
        where
            M: Manager<Terminal = T>,
            M::InnerNode: HasLevel,
        {
            match manager.get_node(&edge) {
                Node::Inner(node) => {
                    let edge = node.child(choices.contains(node.level() as usize) as usize);
                    inner(manager, edge, choices)
                }
                Node::Terminal(t) => t.borrow().clone(),
            }
        }

        inner(manager, edge.borrowed(), &choices)
    }
}

impl<F: Function, T: Tag> DotStyle<T> for MTBDDFunction<F> {}
