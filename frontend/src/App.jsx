// App.jsx - owner: Abheetha
import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./auth/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Layout from "./components/Layout";
import Login from "./pages/Login";
import Register from "./pages/Register";
import Submit from "./pages/Submit";
import Search from "./pages/Search";
import JobTracker from "./pages/JobTracker";
import Results from "./pages/Results";
import History from "./pages/History";
import Compare from "./pages/Compare";
import ModelInfo from "./pages/ModelInfo";
import Admin from "./pages/Admin";

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route element={<ProtectedRoute />}>
          <Route element={<Layout />}>
            <Route path="/submit" element={<Submit />} />
            <Route path="/search" element={<Search />} />
            <Route path="/jobs" element={<JobTracker />} />
            <Route path="/jobs/:id" element={<JobTracker />} />
            <Route path="/results" element={<Results />} />
            <Route path="/history" element={<History />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/model" element={<ModelInfo />} />
            <Route path="/admin" element={<Admin />} />
          </Route>
        </Route>
        <Route path="*" element={<Navigate to="/submit" replace />} />
      </Routes>
    </AuthProvider>
  );
}
