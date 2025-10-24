import React from 'react';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import Hero from './components/Hero';
import Services from './components/Services';
import Gallery from './components/Gallery';
import Pricing from './components/Pricing';
import Contact from './components/Contact';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="App">
      <Toaster position="top-right" toastOptions={{ duration: 4000, style: { background: '#ec4899', color: '#fff' } }} />
      <Header />
      <Hero />
      <Services />
      <Gallery />
      <Pricing />
      <Contact />
      <Footer />
    </div>
  );
}

export default App;
