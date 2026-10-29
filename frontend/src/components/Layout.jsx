import { Link, Outlet } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

const links = [
  ["/submit", "Submit"], ["/search", "Search"], ["/jobs", "Jobs"],
  ["/results", "Results"], ["/history", "History"], ["/compare", "Compare"],
  ["/model", "Model info"], ["/admin", "Admin"],
];

export default function Layout() {
  const { logout } = useAuth();
  return (
    <>
      <nav style={{ display: "flex", gap: 16, padding: "12px 20px", borderBottom: "1px solid #ddd", alignItems: "center" }}>
        {links.map(([to, label]) => <Link key={to} to={to}>{label}</Link>)}
        <button onClick={logout} style={{ marginLeft: "auto" }}>Log out</button>
      </nav>
      <main style={{ padding: 20 }}><Outlet /></main>
    </>
  );
}
