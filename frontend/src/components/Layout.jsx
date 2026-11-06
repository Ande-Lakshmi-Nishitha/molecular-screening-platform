import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

const links = [
  ["/submit", "Submit"], ["/search", "Search"], ["/jobs", "Jobs"],
  ["/results", "Results"], ["/history", "History"], ["/compare", "Compare"],
  ["/model", "Model info"], ["/admin", "Admin"],
];

// The header and the page card are styled in styles.css. NavLink marks the current page (aria-current="page").
export default function Layout() {
  const { logout } = useAuth();
  return (
    <>
      <header className="app-header">
        <nav className="app-nav" aria-label="Main">
          <span className="app-brand">Molecular Screening</span>
          <div className="app-links">
            {links.map(([to, label]) => <NavLink key={to} to={to}>{label}</NavLink>)}
          </div>
          <button type="button" className="ghost" onClick={logout}>Log out</button>
        </nav>
      </header>
      <main className="app-main"><Outlet /></main>
    </>
  );
}
