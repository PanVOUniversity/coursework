import React from 'react';
import { motion } from 'framer-motion';
import './styles.css';

const Gallery = () => {
  const images = [
    'https://images.unsplash.com/photo-1562322140-8baeececf3df?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    'https://images.unsplash.com/photo-1560066984-138dadb4c035?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    'https://images.unsplash.com/photo-1522337660859-02fbefca4702?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    'https://images.unsplash.com/photo-1521590832167-7bcbfaa6381f?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    'https://images.unsplash.com/photo-1595475884562-073c0c879de4?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
    'https://images.unsplash.com/photo-1519699047748-de8e457a634e?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
  ];

  return (
    <section id="gallery" className="gallery">
      <div className="container">
        <motion.div className="section-header" initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }}>
          <h2 className="section-title">Галерея работ</h2>
        </motion.div>

        <div className="gallery-grid">
          {images.map((image, index) => (
            <motion.div key={index} className="gallery-item" initial={{ opacity: 0, scale: 0.8 }} whileInView={{ opacity: 1, scale: 1 }} 
              transition={{ delay: index * 0.1 }} viewport={{ once: true }} whileHover={{ scale: 1.05 }}>
              <img src={image} alt={`Work ${index + 1}`} />
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default Gallery;
