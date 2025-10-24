import React from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import Hero from './components/Hero';
import Features from './components/Features';
import Properties from './components/Properties';
import Services from './components/Services';
import About from './components/About';
import Contact from './components/Contact';
import Footer from './components/Footer';
import './App.css';

function App() {
  return (
    <div className="App">
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#2d5a27',
            color: '#fff',
          },
        }}
      />
      <Header />
      <Hero />
      <Features />
      <Properties />
      <Services />
      <About />
      <Contact />
      <Footer />
    </div>
  );
}

export default App;
