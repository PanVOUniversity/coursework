import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useInView } from 'react-intersection-observer';
import { FaBed, FaBath, FaCar, FaMapMarkerAlt, FaEye, FaHeart } from 'react-icons/fa';
import './Properties.css';

const Properties = () => {
  const [ref, inView] = useInView({
    threshold: 0.1,
    triggerOnce: true
  });

  const [favorites, setFavorites] = useState(new Set());

  const properties = [
    {
      id: 1,
      title: 'Вилла в Palm Jumeirah',
      location: 'Palm Jumeirah, Dubai',
      price: '$2,500,000',
      type: 'Продажа',
      bedrooms: 5,
      bathrooms: 6,
      parking: 3,
      area: '450 м²',
      image: 'https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      gallery: [
        'https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
      ]
    },
    {
      id: 2,
      title: 'Апартаменты в Downtown',
      location: 'Downtown Dubai',
      price: '$8,500/месяц',
      type: 'Аренда',
      bedrooms: 3,
      bathrooms: 2,
      parking: 1,
      area: '180 м²',
      image: 'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      gallery: [
        'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
      ]
    },
    {
      id: 3,
      title: 'Пентхаус в Marina',
      location: 'Dubai Marina',
      price: '$1,800,000',
      type: 'Продажа',
      bedrooms: 4,
      bathrooms: 4,
      parking: 2,
      area: '320 м²',
      image: 'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
      gallery: [
        'https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1545324418-cc1a3fa10c00?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80',
        'https://images.unsplash.com/photo-1564013799919-ab600027ffc6?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=80'
      ]
    }
  ];

  const toggleFavorite = (id) => {
    setFavorites(prev => {
      const newFavorites = new Set(prev);
      if (newFavorites.has(id)) {
        newFavorites.delete(id);
      } else {
        newFavorites.add(id);
      }
      return newFavorites;
    });
  };

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
        delayChildren: 0.1
      }
    }
  };

  const itemVariants = {
    hidden: { y: 50, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.6,
        ease: "easeOut"
      }
    }
  };

  return (
    <section id="properties" className="properties">
      <div className="container">
        <motion.div
          className="section-header"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h2 className="section-title">Премиальные объекты</h2>
          <p className="section-subtitle">
            Эксклюзивная подборка лучших объектов недвижимости в Дубае
          </p>
        </motion.div>

        <motion.div
          className="properties-grid"
          ref={ref}
          variants={containerVariants}
          initial="hidden"
          animate={inView ? "visible" : "hidden"}
        >
          {properties.map((property) => (
            <motion.div
              key={property.id}
              className="property-card"
              variants={itemVariants}
              whileHover={{ y: -10 }}
            >
              <div className="property-image-container">
                <img
                  src={property.image}
                  alt={property.title}
                  className="property-image"
                />
                <div className="property-badge">{property.type}</div>
                <div className="property-actions">
                  <motion.button
                    className="action-btn"
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <FaEye />
                  </motion.button>
                  <motion.button
                    className={`action-btn ${favorites.has(property.id) ? 'favorited' : ''}`}
                    onClick={() => toggleFavorite(property.id)}
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                  >
                    <FaHeart />
                  </motion.button>
                </div>
                <div className="property-overlay" />
              </div>

              <div className="property-content">
                <h3 className="property-title">{property.title}</h3>
                <p className="property-location">
                  <FaMapMarkerAlt />
                  {property.location}
                </p>

                <div className="property-details">
                  <div className="detail-item">
                    <FaBed />
                    <span>{property.bedrooms} спален</span>
                  </div>
                  <div className="detail-item">
                    <FaBath />
                    <span>{property.bathrooms} ванных</span>
                  </div>
                  <div className="detail-item">
                    <FaCar />
                    <span>{property.parking} гаража</span>
                  </div>
                  <div className="detail-item">
                    <span className="area">{property.area}</span>
                  </div>
                </div>

                <div className="property-footer">
                  <div className="property-price">{property.price}</div>
                  <motion.button
                    className="btn btn-primary btn-sm"
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    Подробнее
                  </motion.button>
                </div>
              </div>
            </motion.div>
          ))}
        </motion.div>

        <motion.div
          className="properties-cta"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true }}
        >
          <motion.button
            className="btn btn-outline"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Посмотреть все объекты
          </motion.button>
        </motion.div>
      </div>
    </section>
  );
};

export default Properties;
